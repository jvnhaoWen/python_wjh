import os
import cv2
import time
import threading
from queue import Queue, Empty
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFileDialog, QSlider, QMessageBox)
from qfluentwidgets import PushButton, PrimaryPushButton, SubtitleLabel, ComboBox, CheckBox

# 尝试导入 ultralytics
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class InferenceThread(QThread):
    """专门负责后台推理的线程"""
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.input_queue = Queue(maxsize=1)  # 只保留最新的一帧，防止积压
        self.running = True
        self.yolo_model = None
        self.do_yolo = False

    def run(self):
        while self.running:
            try:
                # 获取待处理的帧，超时 0.1s 检查一次 running 状态
                frame = self.input_queue.get(timeout=0.1)
                
                if self.do_yolo and HAS_YOLO:
                    if self.yolo_model is None:
                        self.yolo_model = YOLO('yolov8n.pt')
                    
                    # 执行推理
                    results = self.yolo_model(frame, verbose=False)
                    # 绘制结果
                    processed_frame = results[0].plot(line_width=2, font_size=1, labels=True, boxes=True)
                    self.result_ready.emit(processed_frame)
                else:
                    # 如果没开启 YOLO，直接原样返回（或者不发送）
                    self.result_ready.emit(frame)
                
                self.input_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"推理线程出错: {e}")

    def stop(self):
        self.running = False


class PlayThread(QThread):
    """视频播放线程 - 负责读取和显示"""
    frame_ready = pyqtSignal(np.ndarray)
    pos_changed = pyqtSignal(int)
    play_end = pyqtSignal()

    def __init__(self, path):
        super().__init__()
        self.cap = cv2.VideoCapture(path)
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.running = True
        self.paused = False
        self.speed = 1.0
        
        # 结果缓存，用于平滑显示
        self.latest_processed_frame = None
        self.lock = threading.Lock()

    def update_processed_frame(self, frame):
        with self.lock:
            self.latest_processed_frame = frame

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
                
            start_time = time.time()
            ok, frame = self.cap.read()
            if not ok:
                self.play_end.emit()
                break

            # 将当前帧交给推理线程（如果推理线程忙，它会自动丢弃旧帧）
            # 这里由外部 Widget 控制推理线程的输入
            self.frame_ready.emit(frame)

            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.pos_changed.emit(pos)
            
            # 严格控制播放速度，不受推理影响
            elapsed = time.time() - start_time
            delay = (1.0 / (self.fps * self.speed)) - elapsed
            if delay > 0:
                time.sleep(delay)

        self.cap.release()

    def seek(self, pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

    def stop(self):
        self.running = False


class PlaybackWidget(QWidget):
#（UI + 调度）
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('PlaybackInterface')
        self.play_thread = None
        
        # 初始化推理线程
        self.infer_thread = InferenceThread()
        self.infer_thread.result_ready.connect(self.on_inference_result)
        self.infer_thread.start()

        self.current_display_frame = None

        layout = QHBoxLayout(self)

        # 左侧画面+进度条
        left = QVBoxLayout()
        self.screen = QLabel('请打开视频文件')
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setStyleSheet('background:black;border-radius:10px;color:white;')
        self.screen.setMinimumSize(480, 360)
        left.addWidget(self.screen)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.on_seek)
        left.addWidget(self.slider)

        btns = QHBoxLayout()
        self.btn_play = PrimaryPushButton('播放')
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        btns.addWidget(self.btn_play)
        left.addLayout(btns)

        # 右侧控制
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignTop)

        self.btn_open = PushButton('打开视频文件')
        self.btn_open.clicked.connect(self.open_file)
        right.addWidget(self.btn_open)

        right.addSpacing(10)
        right.addWidget(QLabel('播放速度:'))
        self.speed_box = ComboBox()
        self.speed_box.addItems(['0.5x', '1.0x', '1.5x', '2.0x'])
        self.speed_box.setCurrentIndex(1)
        right.addWidget(self.speed_box)

        right.addSpacing(10)
        right.addWidget(SubtitleLabel('检测功能'))
        self.ck_yolo = CheckBox('YOLOv8 异步智能识别')
        
        if not HAS_YOLO:
            self.ck_yolo.setEnabled(False)
            self.ck_yolo.setText('YOLOv8 (未安装)')

        right.addWidget(self.ck_yolo)

        layout.addLayout(left, 3)
        layout.addLayout(right, 1)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择视频', '', 'Video (*.mp4 *.avi *.mkv)')
        if not path:
            return
        if self.play_thread and self.play_thread.isRunning():
            self.play_thread.stop()
            self.play_thread.wait()

        self.play_thread = PlayThread(path)
        self.play_thread.frame_ready.connect(self.process_new_frame)
        self.play_thread.pos_changed.connect(self.update_slider)
        self.play_thread.play_end.connect(self.on_end)

        self.slider.setRange(0, self.play_thread.total)
        self.slider.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.screen.setText('已加载: ' + os.path.basename(path))

    def process_new_frame(self, frame):
        """处理新读取的帧"""
        # 同步开关状态
        self.infer_thread.do_yolo = self.ck_yolo.isChecked()
        
        # 如果没开启 YOLO，直接显示原图
        if not self.ck_yolo.isChecked():
            self.show_frame(frame)
        else:
            # 开启了 YOLO，将帧送入推理队列（如果队列满了，put_nowait 会抛出异常，我们直接忽略，实现自动丢帧）
            try:
                self.infer_thread.input_queue.put_nowait(frame)
            except:
                pass
            
            # 如果之前有处理好的帧，先显示处理好的，保证画面不黑屏
            if self.current_display_frame is not None:
                self.show_frame(self.current_display_frame)
            else:
                self.show_frame(frame)

    def on_inference_result(self, processed_frame):
        """推理线程算完一帧后的回调"""
        self.current_display_frame = processed_frame
        # 只有在开启 YOLO 时才通过这个回调刷新画面
        if self.ck_yolo.isChecked():
            self.show_frame(processed_frame)

    def toggle_play(self):
        if not self.play_thread:
            return
        if not self.play_thread.isRunning():
            speeds = [0.5, 1.0, 1.5, 2.0]
            self.play_thread.speed = speeds[self.speed_box.currentIndex()]
            self.play_thread.start()
            self.btn_play.setText('暂停')
        else:
            self.play_thread.paused = not self.play_thread.paused
            self.btn_play.setText('继续' if self.play_thread.paused else '暂停')

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        img = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.screen.setPixmap(QPixmap.fromImage(img).scaled(
            self.screen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_slider(self, pos):
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)

    def on_seek(self, pos):
        if self.play_thread:
            self.play_thread.seek(pos)

    def on_end(self):
        self.btn_play.setText('播放')

    def closeEvent(self, event):
        """窗口关闭时停止所有线程"""
        if self.play_thread:
            self.play_thread.stop()
            self.play_thread.wait()
        self.infer_thread.stop()
        self.infer_thread.wait()
        super().closeEvent(event)
