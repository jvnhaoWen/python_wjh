import os
import cv2
import platform
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QMessageBox, QProgressBar)
from qfluentwidgets import PushButton, PrimaryPushButton, SubtitleLabel, ComboBox

class ConvertThread(QThread):
    """视频转换线程：负责视频格式转换和分辨率调整"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, output_path, width, height):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.width = width
        self.height = height
        self.running = True

    def _open_capture(self):
        candidates = []
        if platform.system() == 'Darwin':
            candidates.append(cv2.VideoCapture(self.input_path, cv2.CAP_AVFOUNDATION))
        candidates.append(cv2.VideoCapture(self.input_path))

        for cap in candidates:
            if cap.isOpened():
                return cap
            cap.release()
        return candidates[-1]

    def _create_writer(self, fps):
        ext = os.path.splitext(self.output_path)[1].lower()
        codec_candidates = {
            '.mp4': ['mp4v', 'avc1', 'H264'],
            '.mov': ['mp4v', 'avc1', 'MJPG'],
            '.avi': ['MJPG', 'XVID']
        }.get(ext, ['mp4v', 'MJPG', 'XVID'])

        for codec in codec_candidates:
            writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*codec),
                fps,
                (self.width, self.height)
            )
            if writer.isOpened():
                return writer
            writer.release()
        return None

    def run(self):
        cap = None
        out = None
        try:
            cap = self._open_capture()
            if not cap.isOpened():
                self.error.emit("无法打开输入视频文件")
                return

            ok, first_frame = cap.read()
            if not ok:
                self.error.emit("输入视频无法被解码，可能是当前编码格式不受支持")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            out = self._create_writer(fps)
            if out is None:
                self.error.emit("无法创建输出视频文件，当前系统不支持所选输出编码")
                return

            count = 0
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break

                resized_frame = cv2.resize(frame, (self.width, self.height))
                out.write(resized_frame)

                count += 1
                if count % 10 == 0 and total_frames > 0:
                    self.progress.emit(int(count * 100 / total_frames))

            if self.running:
                self.progress.emit(100)
                self.finished.emit(self.output_path)
            else:
                if os.path.exists(self.output_path):
                    os.remove(self.output_path)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

    def stop(self):
        self.running = False

class ConverterWidget(QWidget):
    """视频格式转换界面"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ConverterInterface')
        self.convert_thread = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        layout.addWidget(SubtitleLabel('视频格式转换与分辨率调整'))

        # 文件选择
        file_layout = QHBoxLayout()
        self.lbl_file = QLabel('请选择要转换的视频文件')
        self.lbl_file.setStyleSheet('color: gray;')
        self.btn_select = PushButton('选择文件')
        self.btn_select.clicked.connect(self.select_file)
        file_layout.addWidget(self.lbl_file)
        file_layout.addWidget(self.btn_select)
        layout.addLayout(file_layout)

        # 转换设置
        settings_layout = QHBoxLayout()
        
        # 目标格式
        format_layout = QVBoxLayout()
        format_layout.addWidget(QLabel('目标格式:'))
        self.format_box = ComboBox()
        self.format_box.addItems(['.mp4', '.mov', '.avi'])
        format_layout.addWidget(self.format_box)
        settings_layout.addLayout(format_layout)

        # 目标分辨率
        res_layout = QVBoxLayout()
        res_layout.addWidget(QLabel('目标分辨率:'))
        self.res_box = ComboBox()
        self.res_box.addItems(['480p (640x480)', '720p (1280x720)', '1080p (1920x1080)'])
        res_layout.addWidget(self.res_box)
        settings_layout.addLayout(res_layout)
        
        layout.addLayout(settings_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 转换按钮
        self.btn_convert = PrimaryPushButton('开始转换')
        self.btn_convert.clicked.connect(self.start_conversion)
        self.btn_convert.setEnabled(False)
        layout.addWidget(self.btn_convert)
        
        layout.addStretch(1)

        self.input_path = ""

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Video (*.mp4 *.mov *.avi *.mkv *.m4v)')
        if path:
            self.input_path = path
            self.lbl_file.setText(os.path.basename(path))
            self.lbl_file.setStyleSheet('color: black;')
            self.btn_convert.setEnabled(True)

    def start_conversion(self):
        if not self.input_path: return
        
        output_dir = QFileDialog.getExistingDirectory(self, '选择保存目录')
        if not output_dir: return
        
        # 构造输出路径
        ext = self.format_box.currentText()
        base_name = os.path.splitext(os.path.basename(self.input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_converted{ext}")
        
        # 获取分辨率
        res_map = {0: (640, 480), 1: (1280, 720), 2: (1920, 1080)}
        w, h = res_map[self.res_box.currentIndex()]
        
        # 禁用 UI
        self.btn_convert.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # 启动线程
        self.convert_thread = ConvertThread(self.input_path, output_path, w, h)
        self.convert_thread.progress.connect(self.progress_bar.setValue)
        self.convert_thread.finished.connect(self.on_finished)
        self.convert_thread.error.connect(self.on_error)
        self.convert_thread.start()

    def on_finished(self, path):
        self.progress_bar.setVisible(False)
        self.btn_convert.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.convert_thread = None
        QMessageBox.information(self, "成功", f"视频转换完成！\n保存路径: {path}")

    def on_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_convert.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.convert_thread = None
        QMessageBox.critical(self, "错误", f"转换失败: {msg}")
