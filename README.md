# 视频流采集与处理系统

这是我的第一阶段实训项目，用 Python 和 PyQt5 做的一个视频流处理小工具。
界面用了 qfluentwidgets，看起来比较现代一点。

## 实现了什么功能

1. **实时采集**
   - 能打开电脑摄像头看画面
   - 能录制成 MP4 视频
   - 顺便加了人脸、行人和车辆的框框检测（用的 OpenCV 自带的 Haar 和 HOG 模型）

2. **视频回放**
   - 能打开本地的 MP4 或 AVI 视频播放
   - 有个进度条可以拖动
   - 可以选 0.5x 到 2.0x 的播放速度
   - 播放的时候也能开检测框

3. **格式转换**
   - 简单的把 MP4 转成 AVI 格式
   - 有个进度条显示转换了多少

## 怎么运行

先装一下需要的库（最好建个虚拟环境）：

```bash
pip install -r requirements.txt
```

然后直接跑主程序就行了：

```bash
python main.py
```

## 项目结构

```
.
├── main.py                # 程序的入口，就是主窗口
├── widgets/               # 里面是三个主要界面的代码
│   ├── capture_widget.py  # 采集和录制功能
│   ├── playback_widget.py # 视频回放功能
│   └── converter_widget.py# 格式转换功能
├── docs/                  # 老师发的那些需求和测试文档
├── requirements.txt       # 需要装的 Python 库
└── README.md              # 就是这个说明文件
```

## 遇到的卡帧问题以及解决（使用queue）：
```
t=0ms   队列：[frame1] → 被取走 → 开始推理 frame1
t=33ms  队列：[frame2]
t=66ms  队列：[frame3]（覆盖frame2）
t=99ms  队列：[frame4]（覆盖frame3）
t=100ms frame1 推理完成
        ↓
        从队列取 frame4（最新的）
```






启动使用

export QT_QPA_PLATFORM_PLUGIN_PATH=$(python -c "import os, PyQt5; print(os.path.join(os.path.dirname(PyQt5.__file__), 'Qt5', 'plugins', 'platforms'))")
python main.py
