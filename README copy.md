# ROS 人跟随（Jetson + RealSense + BUNKER MINI）

本仓库用于在 **Jetson Xavier（JetPack4 / Ubuntu18.04）+ ROS Melodic + RealSense D435i + BUNKER MINI** 上实现“人跟随”。

核心链路：

`相机压缩图 /camera/color/image_raw/compressed` → **检测（Python3，rosbridge）** 发布 `/person_follow/target` → `follow_controller_node.py` 输出 `/cmd_vel` → `topic_tools relay` 到 `/smoother_cmd_vel` → 底盘驱动。

> 说明：为避免 ROS Melodic 的 **Python2** 与检测侧 **Python3** 混环境，本项目检测端使用 `roslibpy` 通过 `rosbridge_websocket` 与 ROS 通信。

---

## 项目结构（关键文件/目录）

- `person_detect_rosbridge_yolo38.py`：检测端（Python3），支持
  - **ultralytics**（`.pt/.onnx`）推理
  - **TensorRT**（`.engine`）推理：`--backend trt`
- `catkin_ws/src/person_follow/`：跟随控制 ROS 包（Jetson 上）
- `bunker_ws/`：BUNKER MINI 底盘 ROS 驱动工作空间（Jetson 上）

仓库内还保留了你历史记录用的 `README.txt` / `README`（**不会被覆盖**）。GitHub 推荐阅读本文档 `README.md`。

---

## 环境要求

### Jetson（推荐环境）

- **Ubuntu 18.04**
- **ROS Melodic**
- **CUDA 10.2 / TensorRT 8.2.x**（JetPack4 常见）
- RealSense：`realsense2_camera`
- BUNKER MINI 底盘驱动：`bunker_bringup`

### Python 依赖（Jetson）

检测脚本依赖（系统 Python3）：

- `roslibpy`
- `opencv-python`（Jetson 常用是系统 OpenCV，也可用 pip 版本）
- `numpy`
- TensorRT backend 额外需要：`tensorrt`、`pycuda`

---

## 部署（Jetson 端）

### 1) 通用：每个 ROS 终端建议先做的事

```bash
conda deactivate
unset PYTHONPATH
source /opt/ros/melodic/setup.bash
```

### 2) CAN / 底盘准备（BUNKER MINI：500k）

```bash
sudo modprobe gs_usb
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 500000
ip -details link show can0
```

可选验证：

```bash
candump can0
```

---

## 运行（分终端启动，便于定位问题）

> 推荐**分开启动**，不要强行把所有东西塞进一个总 `launch`，这样更容易排查网络、环境、设备问题。

### 1) 底盘终端

```bash
source ~/bunker_ws/devel/setup.bash
roslaunch bunker_bringup bunker_robot_base.launch is_bunker_mini:=true
```

### 2) 跟随控制终端（慢稳档示例）

```bash
source ~/catkin_ws/devel/setup.bash
rosrun person_follow follow_controller_node.py \
  _image_width:=640 \
  _max_lin:=0.08 _max_ang:=0.10 \
  _kp_lin:=0.22 _kp_ang:=0.10 \
  _desired_h:=380 \
  _target_timeout:=0.35
```

### 3) relay 终端（/cmd_vel → /smoother_cmd_vel）

```bash
rosrun topic_tools relay /cmd_vel /smoother_cmd_vel
```

### 4) rosbridge 终端（Python3 检测端通过 websocket 连接）

```bash
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
```

### 5) 相机终端（RealSense）

低延迟/低带宽示例（可按需调整）：

```bash
roslaunch realsense2_camera rs_camera.launch color_width:=424 color_height:=240 color_fps:=15
```

（你也可以不显式传 `color_fps`，但建议用 `rostopic hz` 以实际为准。）

### 6) 检测终端（TensorRT engine，推荐）

先确保系统 Python3 安装依赖：

```bash
python3 -m pip install --user roslibpy
```

运行（TensorRT）：

```bash
python3 /home/h/person_detect_rosbridge_yolo38.py \
  --backend trt \
  --weights /home/h/models/yolov8n_fp16.engine \
  --conf 0.25 \
  --ros-host 127.0.0.1 --ros-port 9090 \
  --image-topic /camera/color/image_raw/compressed \
  --target-topic /person_follow/target \
  --classes 0
```

定位瓶颈（推荐开启一次）：

```bash
python3 /home/h/person_detect_rosbridge_yolo38.py \
  --backend trt \
  --weights /home/h/models/yolov8n_fp16.engine \
  --conf 0.25 \
  --ros-host 127.0.0.1 --ros-port 9090 \
  --profile --profile-interval 1.0
```

---

## 常用排查命令

### 话题频率

```bash
rostopic hz /camera/color/image_raw/compressed
rostopic hz /person_follow/target
```

### 查看检测输出

```bash
rostopic echo /person_follow/target
```

### 查看控制输出

```bash
rostopic echo /cmd_vel
rostopic echo /smoother_cmd_vel
```

---

## 常见问题（FAQ）

### 1) TensorRT `invalid resource handle` / `no currently active context`

这是典型的 **CUDA context 与多线程**问题。当前 `person_detect_rosbridge_yolo38.py` 的 TensorRT backend 已做了：

- 使用 **primary CUDA context**
- 在 worker 线程推理时对 context 进行 `push()/pop()`
- TensorRT 的 execution context/stream/buffer 采用 **worker 线程懒初始化**

如仍异常，优先确认：

- `trtexec --loadEngine ...` 是否能稳定跑通
- 不要在混乱的 venv/conda 环境里运行检测端（建议用系统 `python3`）

### 2) 端到端帧率低

优先看：

- `rostopic hz /camera/color/image_raw/compressed` 是否够高
- 检测端 `--profile` 输出里是 `imdecode` 慢还是 `trt_infer` 慢
- JPEG 数据包 `avg_b64_kb` 是否过大（可降低 `jpeg_quality` 或降低分辨率）

---

## 许可证

如需开源发布，请在此补充 License（MIT/Apache-2.0 等）。

