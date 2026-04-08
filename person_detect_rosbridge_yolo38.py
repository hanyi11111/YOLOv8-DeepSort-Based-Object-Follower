#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import os
import sys
import time
import threading
import importlib

import cv2
import numpy as np
import roslibpy


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


def _nms_xyxy(boxes, scores, iou_thresh):
    """boxes: (N,4) xyxy, scores: (N,)"""
    if boxes.shape[0] == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_r - inter + 1e-6
        iou = inter / union
        idxs = rest[iou <= iou_thresh]
    return keep


class TrtYoloV8Engine(object):
    """
    TensorRT 推理（YOLOv8 导出的典型 ONNX/engine）：
    - 输入 binding: images, 1x3x640x640
    - 输出 binding: output0, 1x84x8400（4 bbox + 80 cls，无 NMS）
    预处理：简单 resize 到 640（若导出使用 letterbox，可后续再对齐）。
    """
    def __init__(self, engine_path, input_name="images", output_name="output0", infer_size=640):
        import tensorrt as trt  # noqa
        import pycuda.driver as cuda  # noqa
        # Use primary context to avoid conflicts with CUDA runtime (TensorRT).
        import pycuda.autoprimaryctx as autoprimaryctx  # noqa: initializes primary CUDA context

        self._cuda = cuda
        self._trt = trt
        # PyCUDA contexts are thread-local. Our pipeline runs inference in a worker thread,
        # so we must push/pop the context around CUDA calls in that thread.
        self._pycuda_ctx = getattr(autoprimaryctx, "context", None)
        if self._pycuda_ctx is None:
            raise RuntimeError("pycuda.autoprimaryctx.context not found")
        self.infer_size = int(infer_size)
        self.input_name = input_name
        self.output_name = output_name

        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("deserialize_cuda_engine failed: {}".format(engine_path))

        self.input_idx = self.engine.get_binding_index(input_name)
        self.output_idx = self.engine.get_binding_index(output_name)
        if self.input_idx < 0 or self.output_idx < 0:
            raise RuntimeError("binding not found: {} / {}".format(input_name, output_name))

        self.in_shape = tuple(self.engine.get_binding_shape(self.input_idx))
        self.out_shape = tuple(self.engine.get_binding_shape(self.output_idx))
        # Binding dtypes (often FP16 for fp16 engines). Using wrong dtype will corrupt outputs
        # and can make cls logits ~0 -> sigmoid ~= 0.5 everywhere.
        self.in_dtype = trt.nptype(self.engine.get_binding_dtype(self.input_idx))
        self.out_dtype = trt.nptype(self.engine.get_binding_dtype(self.output_idx))
        # IMPORTANT: TensorRT execution context / CUDA stream / buffers are initialized lazily
        # in the thread that calls inference (worker thread). Creating them in main thread and
        # using in another thread can lead to huge stalls or invalid resource handles.
        self.context = None
        self.stream = None
        self.d_in = None
        self.d_out = None
        self.bindings = None
        self.last_profile = {}

        print("[det] TRT engine loaded: {}".format(engine_path), flush=True)
        print("[det] TRT input_shape={} input_dtype={} output_shape={} output_dtype={}".format(
            self.in_shape, str(self.in_dtype), self.out_shape, str(self.out_dtype)
        ), flush=True)

    def _ensure_runtime(self):
        """Create execution context/stream/buffers in current thread."""
        if self.context is not None and self.stream is not None and self.bindings is not None:
            return
        import pycuda.driver as cuda  # noqa
        self._pycuda_ctx.push()
        try:
            self.context = self.engine.create_execution_context()
            h_in = int(np.prod(self.in_shape)) * np.dtype(self.in_dtype).itemsize
            h_out = int(np.prod(self.out_shape)) * np.dtype(self.out_dtype).itemsize
            self.d_in = cuda.mem_alloc(h_in)
            self.d_out = cuda.mem_alloc(h_out)
            self.stream = cuda.Stream()
            # bindings array size = num_bindings
            self.bindings = [0] * int(self.engine.num_bindings)
            self.bindings[self.input_idx] = int(self.d_in)
            self.bindings[self.output_idx] = int(self.d_out)
        finally:
            self._pycuda_ctx.pop()

    def infer_raw(self, img_chw):
        """img_chw: (1,3,640,640) contiguous, dtype matches input binding"""
        import pycuda.driver as cuda  # noqa
        self._ensure_runtime()
        out = np.empty(self.out_shape, dtype=self.out_dtype)
        self._pycuda_ctx.push()
        try:
            t0 = time.perf_counter()
            cuda.memcpy_htod_async(self.d_in, img_chw, self.stream)
            t1 = time.perf_counter()
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            t2 = time.perf_counter()
            cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
            t3 = time.perf_counter()
            self.stream.synchronize()
            t4 = time.perf_counter()
            self.last_profile = {
                "h2d_ms": (t1 - t0) * 1000.0,
                "enqueue_ms": (t2 - t1) * 1000.0,
                "d2h_ms": (t3 - t2) * 1000.0,
                "sync_ms": (t4 - t3) * 1000.0,
                "infer_total_ms": (t4 - t0) * 1000.0,
            }
            return out
        finally:
            self._pycuda_ctx.pop()

    def detect(self, frame_bgr, conf_thres, class_ids, iou_thres=0.45):
        """
        返回 dets: list of dict x1,y1,w,h,conf,cls,name
        坐标映射回原图（简单 resize 假设）。
        """
        t_all0 = time.perf_counter()
        h0, w0 = frame_bgr.shape[:2]
        t_r0 = time.perf_counter()
        rz = cv2.resize(frame_bgr, (self.infer_size, self.infer_size), interpolation=cv2.INTER_LINEAR)
        t_r1 = time.perf_counter()
        img = rz[:, :, ::-1].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]
        # Cast to engine input dtype (fp16 engines usually expect fp16 input)
        if self.in_dtype != np.float32:
            img = img.astype(self.in_dtype, copy=False)
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)

        t_inf0 = time.perf_counter()
        raw = self.infer_raw(img)[0]  # (84, 8400) or strip batch
        t_inf1 = time.perf_counter()
        if raw.ndim == 3:
            raw = raw[0]
        # Convert to float32 for stable postprocess math
        raw = raw.astype(np.float32, copy=False)
        t_post0 = time.perf_counter()
        pred = raw.T  # (8400, 84)
        xywh = pred[:, :4]
        cls_logit = pred[:, 4:]
        # Some exports produce logits; some already output 0..1 probabilities.
        # If values are already in [0,1] range, don't apply sigmoid again.
        cmin = float(cls_logit.min()) if cls_logit.size else 0.0
        cmax = float(cls_logit.max()) if cls_logit.size else 0.0
        if cmin >= 0.0 and cmax <= 1.0:
            cls_prob = cls_logit
        else:
            cls_prob = _sigmoid(cls_logit)

        if class_ids:
            cid_list = list(class_ids)
            sub = cls_prob[:, cid_list]
            scores = sub.max(axis=1)
            local_arg = sub.argmax(axis=1)
            clss = np.array(cid_list, dtype=np.int32)[local_arg]
            mask = scores >= conf_thres
        else:
            clss = np.argmax(cls_prob, axis=1)
            scores = cls_prob[np.arange(cls_prob.shape[0]), clss]
            mask = scores >= conf_thres

        keep_idx = np.where(mask)[0]

        if keep_idx.size == 0:
            return []

        xywh_k = xywh[keep_idx]
        scores_k = scores[keep_idx]
        clss_k = clss[keep_idx]

        cx, cy, bw, bh = xywh_k[:, 0], xywh_k[:, 1], xywh_k[:, 2], xywh_k[:, 3]
        x1 = cx - bw * 0.5
        y1 = cy - bh * 0.5
        x2 = cx + bw * 0.5
        y2 = cy + bh * 0.5
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        scale_x = float(w0) / float(self.infer_size)
        scale_y = float(h0) / float(self.infer_size)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        keep = _nms_xyxy(boxes, scores_k, iou_thres)
        dets = []
        coco_names = {0: "person"}
        for j in keep:
            x1, y1, x2, y2 = boxes[j]
            cid = int(clss_k[j])
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            if w < 2 or h < 2:
                continue
            dets.append({
                "x1": float(x1), "y1": float(y1), "w": w, "h": h,
                "conf": float(scores_k[j]), "cls": cid,
                "name": coco_names.get(cid, ""),
            })
        t_post1 = time.perf_counter()
        t_all1 = time.perf_counter()
        # Merge stage timings for outer profiler display
        lp = dict(self.last_profile) if isinstance(self.last_profile, dict) else {}
        lp.update({
            "resize_ms": (t_r1 - t_r0) * 1000.0,
            "infer_wrap_ms": (t_inf1 - t_inf0) * 1000.0,
            "post_ms": (t_post1 - t_post0) * 1000.0,
            "detect_total_ms": (t_all1 - t_all0) * 1000.0,
            "cls_range": (cmin, cmax),
        })
        self.last_profile = lp
        return dets


def _clear_ultralytics_modules():
    stale = [k for k in sys.modules if k == "ultralytics" or k.startswith("ultralytics.")]
    for k in stale:
        sys.modules.pop(k, None)


def _get_yolo_class(import_mode):
    """
    import_mode:
      - auto: try pip ultralytics first, then DeepSort local ultralytics
      - pip: only pip ultralytics
      - deepsort: only DeepSort local ultralytics
    """
    deepsort_root = os.path.expanduser("~/DeepSort/YOLOv8-DeepSORT-Object-Tracking-main")
    deepsort_detect = os.path.join(deepsort_root, "ultralytics", "yolo", "v8", "detect")

    def import_from_pip():
        _clear_ultralytics_modules()
        sys.path[:] = [p for p in sys.path if "YOLOv8-DeepSORT-Object-Tracking-main" not in p and "/DeepSort/" not in p]
        ultra = importlib.import_module("ultralytics")
        print("[det] ultralytics source=pip path={}".format(getattr(ultra, "__file__", "unknown")), flush=True)
        return ultra.YOLO

    def import_from_deepsort():
        _clear_ultralytics_modules()
        if deepsort_root not in sys.path:
            sys.path.insert(0, deepsort_root)
        if deepsort_detect not in sys.path:
            sys.path.insert(0, deepsort_detect)
        ultra = importlib.import_module("ultralytics")

        # Compatibility shim: newer .pt may reference ultralytics.nn.modules.conv/block/head
        import types
        nn_mod = importlib.import_module("ultralytics.nn.modules")
        if not hasattr(nn_mod, "__path__"):
            nn_mod.__path__ = []  # make it package-like for submodule imports
        for sub in ("conv", "block", "head", "transformer"):
            fullname = "ultralytics.nn.modules." + sub
            if fullname not in sys.modules:
                m = types.ModuleType(fullname)
                m.__dict__.update(nn_mod.__dict__)
                sys.modules[fullname] = m

        print("[det] ultralytics source=deepsort path={}".format(getattr(ultra, "__file__", "unknown")), flush=True)
        return ultra.YOLO

    if import_mode == "pip":
        order = [import_from_pip]
    elif import_mode == "deepsort":
        order = [import_from_deepsort]
    else:
        order = [import_from_pip, import_from_deepsort]

    last_err = None
    for fn in order:
        try:
            return fn()
        except Exception as e:
            last_err = e
            print("[det] ultralytics import failed via {}: {}".format(fn.__name__, e), flush=True)
    raise RuntimeError("Failed to import ultralytics with mode={} err={}".format(import_mode, last_err))
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ros-host", default="127.0.0.1")
    p.add_argument("--ros-port", type=int, default=9090)
    p.add_argument("--image-topic", default="/camera/color/image_raw/compressed")
    p.add_argument("--target-topic", default="/person_follow/target")
    p.add_argument("--backend", choices=["ultralytics", "trt"], default="ultralytics",
                   help="ultralytics=YOLO(.pt/.onnx)；trt=TensorRT .engine（需 pycuda+tensorrt）")
    p.add_argument("--weights", default="yolov8s.pt",
                   help="ultralytics 时为 .pt；--backend trt 时为 .engine 路径")
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    p.add_argument("--print-interval", type=float, default=1.0)
    p.add_argument("--fallback-any", action="store_true",
                   help="若无person，回退到任意类别最大框（先跑通链路用）")
    p.add_argument("--show", action="store_true",
                   help="显示调试窗口，查看框和中心线")
    p.add_argument("--classes", type=int, nargs="*", default=[0],
                   help="仅推理指定类别ID，默认[0]=person")
    p.add_argument("--yolo-import-mode", choices=["auto", "pip", "deepsort"], default="auto",
                   help="ultralytics导入来源：auto/pip/deepsort")
    p.add_argument("--profile", action="store_true",
                   help="打印端到端耗时分解（定位 bottleneck）")
    p.add_argument("--profile-interval", type=float, default=1.0,
                   help="profile 打印间隔（秒）")
    args, _unknown = p.parse_known_args()
    return args


class DetectorBridge:
    def __init__(self, args):
        self.args = args
        self.ros = roslibpy.Ros(host=args.ros_host, port=args.ros_port)
        self.pub = None
        self.sub = None

        self.model = None
        self._trt = None
        if args.backend == "trt":
            engine_path = os.path.expanduser(args.weights)
            self._trt = TrtYoloV8Engine(engine_path)
        else:
            yolo_cls = _get_yolo_class(args.yolo_import_mode)
            self.model = yolo_cls(args.weights)
        self._latest_b64 = None
        self._latest_lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._worker = None
        self._last_log_t = 0.0
        self._window_name = "person_follow_debug"
        # Rolling infer FPS: count every _process_frame call (including early returns).
        self._infer_fps_n = 0
        self._infer_fps_t0 = time.time()
        # Rolling receive FPS (rosbridge callback) + lag
        self._recv_n = 0
        self._recv_bytes = 0
        self._recv_t0 = time.time()
        self._last_recv_t = 0.0
        # Rolling profiling sums (ms) over interval
        self._prof_n = 0
        self._prof_t0 = time.time()
        self._prof_sum = {"b64": 0.0, "imdecode": 0.0, "trt": 0.0, "pub": 0.0, "total": 0.0}

    @staticmethod
    def _cls_name(names_map, cid):
        if not isinstance(names_map, dict):
            return ""
        v = names_map.get(cid, None)
        if v is None:
            v = names_map.get(str(cid), None)
        return str(v).strip().lower() if v is not None else ""

    def _log(self, msg):
        t = time.time()
        if t - self._last_log_t >= self.args.print_interval:
            print(msg, flush=True)
            self._last_log_t = t

    def on_image(self, msg):
        data_b64 = msg.get("data", "")
        if not data_b64:
            return
        if self.args.profile:
            self._recv_n += 1
            self._recv_bytes += len(data_b64)
            self._last_recv_t = time.time()
        # Keep only newest frame and drop stale cached frame.
        with self._latest_lock:
            self._latest_b64 = data_b64

    def _pop_latest_b64(self):
        with self._latest_lock:
            data_b64 = self._latest_b64
            self._latest_b64 = None
            return data_b64

    def _worker_loop(self):
        while not self._stop_evt.is_set():
            data_b64 = self._pop_latest_b64()
            if not data_b64:
                time.sleep(0.002)
                continue
            self._process_frame(data_b64)

    def _process_frame(self, data_b64):
        t0 = time.perf_counter()
        tb64 = 0.0
        timd = 0.0
        ttrt = 0.0
        tpub = 0.0
        try:
            t1 = time.perf_counter()
            jpg = base64.b64decode(data_b64)
            t2 = time.perf_counter()
            tb64 = (t2 - t1) * 1000.0
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            t3 = time.perf_counter()
            timd = (t3 - t2) * 1000.0
            if frame is None:
                return

            vis_frame = None
            if self.args.show:
                vis_frame = frame.copy()
                vh, vw = vis_frame.shape[:2]
                cv2.line(vis_frame, (vw // 2, 0), (vw // 2, vh - 1), (255, 0, 0), 2)
                cv2.line(vis_frame, (0, vh // 2), (vw - 1, vh // 2), (255, 0, 0), 1)

            if self._trt is not None:
                t4 = time.perf_counter()
                dets = self._trt.detect(frame,
                                        conf_thres=self.args.conf,
                                        class_ids=self.args.classes if self.args.classes else None)
                t5 = time.perf_counter()
                ttrt = (t5 - t4) * 1000.0
                names_map = {i: n for i, n in enumerate(
                    ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                     "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                     "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                     "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                     "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                     "toothbrush")
                )}
                for d in dets:
                    cid = d["cls"]
                    d["name"] = self._cls_name(names_map, cid) or d.get("name", "")
                if vis_frame is not None:
                    for d in dets:
                        x1, y1, w, h = d["x1"], d["y1"], d["w"], d["h"]
                        x2, y2 = x1 + w, y1 + h
                        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                        cid = d["cls"]
                        cname = d.get("name", "")
                        is_person = cname in ("person", "pedestrian") or (cname == "" and cid == 0)
                        color = (0, 255, 0) if is_person else (0, 0, 255)
                        label = "{} {:.2f}".format(cname or cid, float(d["conf"]))
                        cv2.rectangle(vis_frame, (x1i, y1i), (x2i, y2i), color, 2)
                        cv2.putText(vis_frame, label, (x1i, max(15, y1i - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                source_input = frame
                if self.args.yolo_import_mode == "deepsort":
                    # Old DeepSort-patched ultralytics expects file path source.
                    source_input = "/tmp/person_follow_frame.jpg"
                    cv2.imwrite(source_input, frame)

                predict_kwargs = {
                    "source": source_input,
                    "conf": self.args.conf,
                    "imgsz": self.args.imgsz,
                    "verbose": False,
                }
                if self.args.classes:
                    predict_kwargs["classes"] = self.args.classes
                # TensorRT engine does not need torch CUDA device checks.
                if not str(self.args.weights).lower().endswith(".engine"):
                    predict_kwargs["device"] = self.args.device
                results = self.model.predict(**predict_kwargs)

                if not results:
                    self._log("[det] empty results")
                    return

                r = results[0]
                names_map = getattr(r, "names", None) or getattr(self.model, "names", {}) or {}

                dets = []
                if hasattr(r, "boxes"):
                    # New ultralytics Results format
                    if r.boxes is not None and len(r.boxes) > 0:
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        clss = r.boxes.cls.cpu().numpy()
                        for box, cf, cid in zip(xyxy, confs, clss):
                            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
                            w = max(0.0, x2 - x1)
                            h = max(0.0, y2 - y1)
                            if w < 2 or h < 2:
                                continue
                            cid = int(cid)
                            cname = self._cls_name(names_map, cid)
                            dets.append({
                                "x1": x1, "y1": y1, "w": w, "h": h,
                                "conf": float(cf), "cls": cid, "name": cname
                            })
                            if vis_frame is not None:
                                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                                is_person = cname in ("person", "pedestrian") or (cname == "" and cid == 0)
                                color = (0, 255, 0) if is_person else (0, 0, 255)
                                label = "{} {:.2f}".format(cname or cid, float(cf))
                                cv2.rectangle(vis_frame, (x1i, y1i), (x2i, y2i), color, 2)
                                cv2.putText(vis_frame, label, (x1i, max(15, y1i - 6)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                else:
                    # Old ultralytics tensor format: Nx6 [x1,y1,x2,y2,conf,cls]
                    pred = r
                    if hasattr(pred, "cpu"):
                        pred = pred.cpu().numpy()
                    for row in pred:
                        if len(row) < 6:
                            continue
                        x1, y1, x2, y2, cf, cid = [float(v) for v in row[:6]]
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        if w < 2 or h < 2:
                            continue
                        cid = int(cid)
                        cname = self._cls_name(names_map, cid)
                        dets.append({
                            "x1": x1, "y1": y1, "w": w, "h": h,
                            "conf": float(cf), "cls": cid, "name": cname
                        })
                        if vis_frame is not None:
                            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
                            is_person = cname in ("person", "pedestrian") or (cname == "" and cid == 0)
                            color = (0, 255, 0) if is_person else (0, 0, 255)
                            label = "{} {:.2f}".format(cname or cid, float(cf))
                            cv2.rectangle(vis_frame, (x1i, y1i), (x2i, y2i), color, 2)
                            cv2.putText(vis_frame, label, (x1i, max(15, y1i - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if not dets:
                if vis_frame is not None:
                    cv2.putText(vis_frame, "no box", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow(self._window_name, vis_frame)
                    cv2.waitKey(1)
                self._log("[det] no box")
                return

            persons = [d for d in dets if d["name"] in ("person", "pedestrian") or (d["name"] == "" and d["cls"] == 0)]

            if persons:
                best = max(persons, key=lambda d: d["h"])
                picked = "person"
            elif self.args.fallback_any:
                best = max(dets, key=lambda d: d["h"])
                picked = "fallback_any"
            else:
                top = sorted(dets, key=lambda d: d["conf"], reverse=True)[:3]
                if vis_frame is not None:
                    cv2.putText(vis_frame, "no person", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow(self._window_name, vis_frame)
                    cv2.waitKey(1)
                self._log("[det] boxes={} but no person top={}".format(
                    len(dets),
                    [(d["cls"], d["name"], round(d["conf"], 3)) for d in top]
                ))
                return

            payload = {
                "id": 0,
                "cx": best["x1"] + best["w"] * 0.5,
                "cy": best["y1"] + best["h"] * 0.5,
                "w": best["w"],
                "h": best["h"],
                "conf": best["conf"],
                "stamp": time.time(),
            }

            if vis_frame is not None:
                x1i = int(best["x1"])
                y1i = int(best["y1"])
                x2i = int(best["x1"] + best["w"])
                y2i = int(best["y1"] + best["h"])
                cxi = int(payload["cx"])
                cyi = int(payload["cy"])
                cv2.rectangle(vis_frame, (x1i, y1i), (x2i, y2i), (0, 255, 255), 3)
                cv2.circle(vis_frame, (cxi, cyi), 5, (0, 255, 255), -1)
                cv2.putText(
                    vis_frame,
                    "picked={} cx={:.1f} h={:.1f}".format(picked, payload["cx"], payload["h"]),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                cv2.imshow(self._window_name, vis_frame)
                cv2.waitKey(1)

            t6 = time.perf_counter()
            self.pub.publish(roslibpy.Message({"data": json.dumps(payload)}))
            t7 = time.perf_counter()
            tpub = (t7 - t6) * 1000.0
            self._log("[det] publish {} cls={} name={} conf={:.2f} h={:.1f}".format(
                picked, best["cls"], best["name"], best["conf"], best["h"]
            ))

        except Exception as e:
            print("on_image error:", e, flush=True)
        finally:
            last_ms = (time.perf_counter() - t0) * 1000.0
            self._infer_fps_n += 1
            now = time.time()
            elapsed = now - self._infer_fps_t0
            if elapsed >= 1.0:
                fps = self._infer_fps_n / elapsed
                print("[det] infer_fps={:.2f} frames_in_window={} last_frame_ms={:.1f}".format(
                    fps, self._infer_fps_n, last_ms), flush=True)
                self._infer_fps_n = 0
                self._infer_fps_t0 = now

            if self.args.profile:
                # recv stats (rosbridge delivery speed)
                r_elapsed = now - self._recv_t0
                if r_elapsed >= float(self.args.profile_interval):
                    recv_fps = self._recv_n / r_elapsed if r_elapsed > 0 else 0.0
                    avg_kb = (self._recv_bytes / max(1, self._recv_n)) / 1024.0
                    lag_ms = (now - self._last_recv_t) * 1000.0 if self._last_recv_t > 0 else -1.0
                    print("[det] recv_fps={:.2f} avg_b64_kb={:.1f} last_recv_lag_ms={:.0f}".format(
                        recv_fps, avg_kb, lag_ms), flush=True)
                    self._recv_n = 0
                    self._recv_bytes = 0
                    self._recv_t0 = now

                # proc breakdown
                self._prof_n += 1
                self._prof_sum["b64"] += tb64
                self._prof_sum["imdecode"] += timd
                self._prof_sum["trt"] += ttrt
                self._prof_sum["pub"] += tpub
                self._prof_sum["total"] += last_ms

                p_elapsed = now - self._prof_t0
                if p_elapsed >= float(self.args.profile_interval):
                    n = max(1, self._prof_n)
                    extra = ""
                    if self._trt is not None and hasattr(self._trt, "last_profile") and isinstance(self._trt.last_profile, dict):
                        lp = self._trt.last_profile
                        extra = " trt_infer={:.1f}ms(sync={:.1f}) resize={:.1f} post={:.1f} cls_range=({:.2f},{:.2f})".format(
                            float(lp.get("infer_total_ms", 0.0)),
                            float(lp.get("sync_ms", 0.0)),
                            float(lp.get("resize_ms", 0.0)),
                            float(lp.get("post_ms", 0.0)),
                            float(lp.get("cls_range", (0.0, 0.0))[0]),
                            float(lp.get("cls_range", (0.0, 0.0))[1]),
                        )
                    print(("[det] ms_avg b64={:.1f} imdecode={:.1f} trt_call={:.1f} pub={:.1f} total={:.1f}" + extra).format(
                        self._prof_sum["b64"] / n,
                        self._prof_sum["imdecode"] / n,
                        self._prof_sum["trt"] / n,
                        self._prof_sum["pub"] / n,
                        self._prof_sum["total"] / n,
                    ), flush=True)
                    self._prof_n = 0
                    for k in self._prof_sum:
                        self._prof_sum[k] = 0.0
                    self._prof_t0 = now

    def run(self):
        print("Connecting rosbridge ws://{}:{} ...".format(self.args.ros_host, self.args.ros_port), flush=True)
        self.ros.run()
        if not self.ros.is_connected:
            raise RuntimeError("rosbridge connect failed")

        self.pub = roslibpy.Topic(self.ros, self.args.target_topic, "std_msgs/String")
        self.sub = roslibpy.Topic(self.ros, self.args.image_topic, "sensor_msgs/CompressedImage")
        self.sub.subscribe(self.on_image)
        self._worker = threading.Thread(target=self._worker_loop, name="detector_worker", daemon=True)
        self._worker.start()

        print("Subscribed:", self.args.image_topic, flush=True)
        print("Publishing:", self.args.target_topic, flush=True)

        try:
            while self.ros.is_connected:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if self.sub:
                    self.sub.unsubscribe()
                if self.pub:
                    self.pub.unadvertise()
            finally:
                self._stop_evt.set()
                if self._worker is not None:
                    self._worker.join(timeout=1.0)
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                self.ros.terminate()


if __name__ == "__main__":
    args = parse_args()
    DetectorBridge(args).run()
