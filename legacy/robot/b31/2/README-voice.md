# 离线语音控制（树莓派）

本模块复用本目录的 `robotmove.py`，通过离线 Vosk 将固定中文口令转为已有 UART 动作码；不修改 STM 固件和 `.dzz` 动作。

## 部署

在树莓派中，本目录位于项目的 `modules` 目录。以下命令在该目录执行：

```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio unzip wget
python3 -m pip install -r requirements-voice.txt
mkdir -p models
wget -O /tmp/vosk-cn.zip https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
unzip /tmp/vosk-cn.zip -d models
python3 voice_control.py --check
python3 voice_control.py
```

需要已连接 USB 麦克风，并确保机器人串口为 `/dev/serial0`。若找不到麦克风，先执行 `arecord -l` 检查设备。

## 支持口令

`前进`、`后退`、`左转`、`右转`、`跳舞`、`左脚踢球`、`右脚踢球`、`抱箱子`、`放箱子`。

一次口令只触发一个现有动作。现有串口协议没有可确认的即时刹停指令，因此“停止”不会被映射为动作；紧急情况请用机器人现有硬件停止/复位方式。不要在搬运、踢球等自主任务正在运行时并行启动本程序。
