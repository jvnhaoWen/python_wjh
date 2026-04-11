import os
import cv2
import time
import platform
from queue import Queue, Empty, Full
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QSlider, QMessageBox)
from qfluentwidgets import PushButton, PrimaryPushButton, SubtitleLabel, ComboBox, CheckBox

# 尝试导入 YOLO 识别库
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

class InferenceThread(QThread):
    """后台推理线程：负责所有 AI 检测逻辑"""
    result_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.input_queue = Queue(maxsize=1)
        self.running = True
        self.yolo_model = None
        self.face_cas = None
        self.car_cas = None
        self.hog = None
        self.switches = {'face': False, 'ped': False, 'car': False, 'yolo': False}
        self.pedestrian_scale = 0.5

    def submit_frame(self, frame):
        """只保留最新一帧，避免重检测任务把旧帧堵在队列里。"""
        try:
            self.input_queue.get_nowait()
        except Empty:
            pass

        try:
            self.input_queue.put_nowait(frame.copy())
        except Full:
            pass

    def run(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                
                # 1. 人脸检测
                if self.switches['face']:
                    if not self.face_cas: self.face_cas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cas.detectMultiScale(gray, 1.3, 5)
                    for (x, y, w, h) in faces: cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # 2. 行人检测
                if self.switches['ped']:
                    if not self.hog:
                        self.hog = cv2.HOGDescriptor()
                        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                    detect_frame = frame
                    scale = 1.0
                    if frame.shape[1] > 960 or frame.shape[0] > 540:
                        scale = self.pedestrian_scale
                        detect_frame = cv2.resize(
                            frame,
                            (0, 0),
                            fx=scale,
                            fy=scale,
                            interpolation=cv2.INTER_LINEAR
                        )

                    rects, _ = self.hog.detectMultiScale(
                        detect_frame,
                        winStride=(8, 8),
                        padding=(8, 8),
                        scale=1.05
                    )
                    inv_scale = 1.0 / scale
                    for (x, y, w, h) in rects:
                        x = int(x * inv_scale)
                        y = int(y * inv_scale)
                        w = int(w * inv_scale)
                        h = int(h * inv_scale)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # 3. 车辆检测
                if self.switches['car']:
                    if not self.car_cas: self.car_cas = cv2.CascadeClassifier('haarcascade_car.xml')
                    if self.car_cas.empty(): self.car_cas = None # 防止加载失败
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cars = self.car_cas.detectMultiScale(gray, 1.1, 3)
                        for (x, y, w, h) in cars: cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # 4. YOLOv8 智能识别
                if self.switches['yolo'] and HAS_YOLO:
                    if not self.yolo_model: self.yolo_model = YOLO('yolov8n.pt')
                    results = self.yolo_model(frame, verbose=False)
                    frame = results[0].plot(line_width=2, font_size=1)
                
                self.result_ready.emit(frame)
            except Empty: continue
            except Exception as e: print(f"推理出错: {e}")

    def stop(self): self.running = False

class PlayThread(QThread):
    """视频读取线程：负责按帧率读取视频文件"""
    frame_ready = pyqtSignal(np.ndarray)
    pos_changed = pyqtSignal(int)
    play_end = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.cap = self._open_capture(path)
        if not self.cap.isOpened():
            self.total = 0
            self.fps = 25
        else:
            self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.running, self.paused, self.speed = True, False, 1.0

    def _open_capture(self, path):
        """macOS 优先尝试 AVFoundation，提升本地 mp4/mov 兼容性。"""
        candidates = []
        if platform.system() == 'Darwin':
            candidates.append(cv2.VideoCapture(path, cv2.CAP_AVFOUNDATION))
        candidates.append(cv2.VideoCapture(path))

        for cap in candidates:
            if cap.isOpened():
                return cap
            cap.release()
        return candidates[-1]

    def run(self):
        if not self.cap.isOpened():
            self.error.emit(f"无法打开视频文件: {os.path.basename(self.path)}")
            return

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            
            start_time = time.time()
            ok, frame = self.cap.read()
            if not ok:
                self.play_end.emit()
                break

            self.frame_ready.emit(frame)
            self.pos_changed.emit(int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
            
            # 控制播放速度
            delay = (1.0 / (self.fps * self.speed)) - (time.time() - start_time)
            if delay > 0: time.sleep(delay)
        self.cap.release()

    def stop(self): self.running = False

class PlaybackWidget(QWidget):
    """主界面：负责 UI 交互和视频处理展示"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('PlaybackInterface')
        self.play_thread = None
        self.infer_thread = InferenceThread()
        self.infer_thread.result_ready.connect(self.update_screen)
        self.infer_thread.start()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        
        # 左侧：视频展示区
        left_layout = QVBoxLayout()
        self.screen = QLabel('请上传本地视频进行处理')
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setStyleSheet('background:black; border-radius:10px; color:white;')
        self.screen.setMinimumSize(480, 360)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        
        btn_layout = QHBoxLayout()
        self.btn_play = PrimaryPushButton('播放')
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_stop = PushButton('停止')
        self.btn_stop.clicked.connect(self.stop_play)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_stop)
        
        left_layout.addWidget(self.screen)
        left_layout.addWidget(self.slider)
        left_layout.addLayout(btn_layout)

        # 右侧：控制面板
        right_layout = QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        
        self.btn_open = PushButton('上传本地视频')
        self.btn_open.clicked.connect(self.open_file)
        
        self.speed_box = ComboBox()
        self.speed_box.addItems(['0.5x', '1.0x', '1.5x', '2.0x'])
        self.speed_box.setCurrentIndex(1)
        
        right_layout.addWidget(self.btn_open)
        right_layout.addSpacing(10)
        right_layout.addWidget(QLabel('播放速度:'))
        right_layout.addWidget(self.speed_box)
        right_layout.addSpacing(10)
        right_layout.addWidget(SubtitleLabel('视频处理功能'))
        
        self.checks = {
            'face': CheckBox('人脸检测'),
            'ped': CheckBox('行人检测'),
            'car': CheckBox('车辆检测'),
            'yolo': CheckBox('YOLOv8 智能识别')
        }
        for cb in self.checks.values(): right_layout.addWidget(cb)
        
        layout.addLayout(left_layout, 3)
        layout.addLayout(right_layout, 1)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Video (*.mp4 *.avi *.mkv *.mov *.m4v)')
        if path:
            self.stop_play()
            self.play_thread = PlayThread(path)
            if not self.play_thread.cap.isOpened():
                QMessageBox.critical(self, "错误", f"无法打开视频文件，请检查格式是否受支持。")
                return

            ok, frame = self.play_thread.cap.read()
            if not ok:
                self.play_thread.cap.release()
                self.play_thread = None
                QMessageBox.critical(
                    self,
                    "错误",
                    "视频文件已选择，但无法解码出有效画面。\n"
                    "如果这是 mov/mp4 文件，建议先到“格式转换”页转成 mp4 后再播放。"
                )
                return
            self.play_thread.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.update_screen(frame)
                
            self.play_thread.frame_ready.connect(self.on_frame_received)
            self.play_thread.pos_changed.connect(self.update_slider_pos)
            self.play_thread.play_end.connect(self.on_end)
            self.play_thread.error.connect(lambda m: QMessageBox.critical(self, "错误", m))
            
            self.slider.setRange(0, self.play_thread.total)
            self.slider.setValue(0)
            self.slider.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.screen.setText(f'已上传: {os.path.basename(path)}')

    def on_frame_received(self, frame):
        # 同步开关状态
        for key, cb in self.checks.items(): self.infer_thread.switches[key] = cb.isChecked()
        self.play_thread.speed = [0.5, 1.0, 1.5, 2.0][self.speed_box.currentIndex()]
        
        if not any(self.infer_thread.switches.values()):
            self.update_screen(frame)
        else:
            self.infer_thread.submit_frame(frame)

    def update_screen(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        img = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.screen.setPixmap(QPixmap.fromImage(img).scaled(self.screen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def toggle_play(self):
        if self.play_thread:
            if not self.play_thread.isRunning():
                self.play_thread.start()
                self.btn_play.setText('暂停')
            else:
                self.play_thread.paused = not self.play_thread.paused
                self.btn_play.setText('继续' if self.play_thread.paused else '暂停')

    def stop_play(self):
        if self.play_thread:
            self.play_thread.stop()
            self.play_thread.wait()
            self.play_thread = None
        self.btn_play.setText('播放')
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.screen.setText('请上传本地视频进行处理')

    def on_slider_moved(self, pos):
        if self.play_thread:
            self.play_thread.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            # 拖动时显示当前帧
            ok, frame = self.play_thread.cap.read()
            if ok: self.update_screen(frame)

    def update_slider_pos(self, pos):
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)

    def on_end(self):
        self.btn_play.setText('播放')
        self.btn_play.setEnabled(True)

    def closeEvent(self, event):
        self.stop_play()
        self.infer_thread.stop()
        self.infer_thread.wait()
        super().closeEvent(event)
