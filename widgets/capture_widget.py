import cv2
import os
import time
import threading
from queue import Queue
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFileDialog, QMessageBox)
from qfluentwidgets import (PushButton, ComboBox, CheckBox,
                             PrimaryPushButton, SubtitleLabel, SpinBox)
import numpy as np

# 尝试导入 ultralytics
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class CaptureThread(QThread):
    """摄像头采集线程"""
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, cam_id=0, width=640, height=480):
        super().__init__()
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.running = True
        self.speed = 1.0
        self._emit_counter = 0
        
        # 异步录制相关
        self.recording = False
        self.write_queue = Queue(maxsize=128)
        self.stop_event = threading.Event()
        self.writer = None
        self.writer_thread = None
        
        # 分段存储相关
        self.segment_duration = 0  # 0 表示不分段，单位：秒
        self.record_start_time = 0
        self.save_dir = ""
        self.base_name = ""

        # 检测开关
        self.do_face = False
        self.do_pedestrian = False
        self.do_car = False
        self.do_yolo = False

        # haar 分类器
        self.face_cas = None
        self.car_cas = None
        self.hog = None
        self.pedestrian_scale = 0.5
        
        # YOLO 模型
        self.yolo_model = None

    def writer_worker(self):
        """后台录制线程逻辑"""
        while not self.stop_event.is_set() or not self.write_queue.empty():
            try:
                frame = self.write_queue.get(timeout=0.1)
                if self.writer:
                    self.writer.write(frame)
                self.write_queue.task_done()
            except:
                continue

    def run(self):
        # 针对不同系统优化摄像头开启
        if os.name == 'nt': # Windows
            cap = cv2.VideoCapture(self.cam_id + cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.cam_id)
            
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.cam_id) # 降级尝试
            
        if not cap.isOpened():
            self.error.emit('打不开摄像头')
            return

        # 设置分辨率和FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        while self.running:
            loop_start = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            # 分段存储逻辑检查
            if self.recording and self.segment_duration > 0:
                if time.time() - self.record_start_time >= self.segment_duration:
                    self.switch_segment()

            # 检测逻辑
            try:
                if self.do_face:
                    if self.face_cas is None:
                        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        if os.path.exists(path):
                            self.face_cas = cv2.CascadeClassifier(path)
                    if self.face_cas:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cas.detectMultiScale(gray, 1.3, 5)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if self.do_pedestrian:
                    if self.hog is None:
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

                if self.do_car:
                    if self.car_cas is None:
                        paths = ['haarcascade_car.xml', cv2.data.haarcascades + 'haarcascade_car.xml']
                        for p in paths:
                            if os.path.exists(p):
                                self.car_cas = cv2.CascadeClassifier(p)
                                break
                    if self.car_cas:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cars = self.car_cas.detectMultiScale(gray, 1.1, 3)
                        for (x, y, w, h) in cars:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # 美化后的 YOLOv8 识别
                if self.do_yolo and HAS_YOLO:
                    if self.yolo_model is None:
                        self.yolo_model = YOLO('yolov8n.pt')
                    
                    results = self.yolo_model(frame, verbose=False)
                    frame = results[0].plot(line_width=2, font_size=1, labels=True, boxes=True)
            except Exception as e:
                print(f"检测出错: {e}")

            # 异步录制入队
            if self.recording:
                try:
                    self.write_queue.put_nowait(frame.copy())
                except:
                    pass # 队列满则丢弃

            emit_frame = True
            if self.speed > 1.0:
                skip_map = {1.5: 2, 2.0: 3}
                interval = skip_map.get(self.speed, max(2, int(round(self.speed))))
                emit_frame = (self._emit_counter % interval == 0)
            self._emit_counter += 1

            if emit_frame:
                self.frame_ready.emit(frame)

            base_delay = 0.01
            if self.speed < 1.0:
                base_delay += (1.0 / self.speed - 1.0) * 0.03

            elapsed = time.time() - loop_start
            delay = base_delay - elapsed
            if delay > 0:
                time.sleep(delay)

        cap.release()
        self.stop_recording_internal()

    def start_record(self, save_dir, duration=0):
        self.save_dir = save_dir
        self.segment_duration = duration
        self.record_start_time = time.time()
        self.base_name = time.strftime('video_%Y%m%d_%H%M%S')
        
        path = os.path.join(self.save_dir, f"{self.base_name}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(path, fourcc, 30, (self.width, self.height))
        
        self.stop_event.clear()
        self.writer_thread = threading.Thread(target=self.writer_worker)
        self.writer_thread.start()
        self.recording = True

    def switch_segment(self):
        """切换到下一个录制分段"""
        # 停止当前写入
        old_writer = self.writer
        self.record_start_time = time.time()
        new_name = time.strftime('video_%Y%m%d_%H%M%S')
        path = os.path.join(self.save_dir, f"{new_name}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        # 快速创建新写入器
        self.writer = cv2.VideoWriter(path, fourcc, 30, (self.width, self.height))
        
        # 释放旧写入器
        if old_writer:
            old_writer.release()

    def stop_record(self):
        self.recording = False
        self.stop_recording_internal()

    def stop_recording_internal(self):
        self.stop_event.set()
        if self.writer_thread:
            self.writer_thread.join()
            self.writer_thread = None
        if self.writer:
            self.writer.release()
            self.writer = None

    def stop(self):
        self.running = False


class CaptureWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('CaptureInterface')
        self.thread = None

        layout = QHBoxLayout(self)

        # 左边放画面
        left = QVBoxLayout()
        self.screen = QLabel('摄像头未开启')
        self.screen.setAlignment(Qt.AlignCenter)
        self.screen.setStyleSheet('background:black;border-radius:10px;color:white;')
        self.screen.setMinimumSize(480, 360)
        left.addWidget(self.screen)

        # 右边放控制
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignTop)

        self.btn_start = PrimaryPushButton('开始采集')
        self.btn_start.clicked.connect(self.toggle)
        right.addWidget(self.btn_start)

        right.addSpacing(10)
        right.addWidget(SubtitleLabel('分辨率设置'))
        self.res_box = ComboBox()
        self.res_box.addItems(['480p (640x480)', '720p (1280x720)', '1080p (1920x1080)'])
        self.res_box.setCurrentIndex(0)
        right.addWidget(self.res_box)

        right.addSpacing(10)
        right.addWidget(SubtitleLabel('采集速度'))
        self.speed_box = ComboBox()
        self.speed_box.addItems(['0.5x', '1.0x', '1.5x', '2.0x'])
        self.speed_box.setCurrentIndex(1)
        right.addWidget(self.speed_box)

        right.addSpacing(10)
        right.addWidget(SubtitleLabel('检测功能'))
        self.ck_face = CheckBox('人脸检测')
        self.ck_ped = CheckBox('行人检测')
        self.ck_car = CheckBox('车辆检测')
        self.ck_yolo = CheckBox('YOLOv8 智能识别')
        
        if not HAS_YOLO:
            self.ck_yolo.setEnabled(False)
            self.ck_yolo.setText('YOLOv8 (未安装)')

        right.addWidget(self.ck_face)
        right.addWidget(self.ck_ped)
        right.addWidget(self.ck_car)
        right.addWidget(self.ck_yolo)

        right.addSpacing(10)
        right.addWidget(SubtitleLabel('分段存储 (分钟)'))
        self.spin_duration = SpinBox()
        self.spin_duration.setRange(0, 60)
        self.spin_duration.setValue(0)  # 0 表示不分段
        right.addWidget(self.spin_duration)

        right.addSpacing(10)
        self.btn_rec = PushButton('开始录制 (异步)')
        self.btn_rec.clicked.connect(self.toggle_record)
        self.btn_rec.setEnabled(False)
        right.addWidget(self.btn_rec)

        layout.addLayout(left, 3)
        layout.addLayout(right, 1)

    def toggle(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait()
            self.thread = None
            self.btn_start.setText('开始采集')
            self.btn_rec.setEnabled(False)
            self.res_box.setEnabled(True)
            self.speed_box.setEnabled(True)
            self.screen.setText('摄像头已关闭')
        else:
            # 获取选择的分辨率
            res_map = {0: (640, 480), 1: (1280, 720), 2: (1920, 1080)}
            w, h = res_map[self.res_box.currentIndex()]
            
            self.thread = CaptureThread(0, w, h)
            self.thread.speed = [0.5, 1.0, 1.5, 2.0][self.speed_box.currentIndex()]
            self.thread.frame_ready.connect(self.show_frame)
            self.thread.error.connect(lambda m: QMessageBox.warning(self, '错误', m))
            self.thread.start()
            self.btn_start.setText('停止采集')
            self.btn_rec.setEnabled(True)
            self.res_box.setEnabled(False)
            self.speed_box.setEnabled(False)

    def show_frame(self, frame):
        if self.thread:
            self.thread.do_face = self.ck_face.isChecked()
            self.thread.do_pedestrian = self.ck_ped.isChecked()
            self.thread.do_car = self.ck_car.isChecked()
            self.thread.do_yolo = self.ck_yolo.isChecked()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        img = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
        self.screen.setPixmap(QPixmap.fromImage(img).scaled(
            self.screen.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def toggle_record(self):
        if not self.thread:
            return
        if not self.thread.recording:
            d = QFileDialog.getExistingDirectory(self, '选择保存目录')
            if d:
                duration_min = self.spin_duration.value()
                self.thread.start_record(d, duration_min * 60)
                self.btn_rec.setText('停止录制')
                self.spin_duration.setEnabled(False)
        else:
            self.thread.stop_record()
            self.btn_rec.setText('开始录制 (异步)')
            self.spin_duration.setEnabled(True)
