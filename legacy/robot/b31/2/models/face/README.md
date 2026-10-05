# 树莓派人脸模型

`face.eim` 是 Edge Impulse 的 Linux AARCH64 模型，已从下载目录复制到这里；原文件未改动。

模型标签为：`haowenwan`、`wujuncheng`、`xieyuting`、`yangchengrui`、`zhaobowen`、`none`。

`main_1.py` 使用这个模型识别五位已训练人员；同一身份连续 2 帧置信度至少为 0.65 才会进入下一任务。中文播报名在 `main_1.py` 的 `FACE_NAMES` 中设置。

在树莓派项目目录执行：

```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
python3 -m pip install --user --break-system-packages edge_impulse_linux
chmod +x models/face/face.eim
cd ../..
python3 main_1.py face
```

部署前先检查：

```bash
uname -m                    # 必须为 aarch64
./models/face/face.eim --print-info
```
