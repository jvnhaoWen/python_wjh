import os
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QFileDialog, QMessageBox, QProgressBar)
from qfluentwidgets import PushButton, PrimaryPushButton, SubtitleLabel


class ConvertThread(QThread):
    progress = pyqtSignal(int)
    done = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, src, dst):
        super().__init__()
        self.src = src
        self.dst = dst

    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            self.error.emit('无法打开文件')
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(self.dst, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(frame)
            i += 1
            if i % 30 == 0 and total > 0:
                self.progress.emit(int(i / total * 100))

        cap.release()
        out.release()
        self.progress.emit(100)
        self.done.emit(self.dst)

#转换，写入，保存
class ConverterWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ConverterInterface')
        self.src_file = ''

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(15)

        layout.addWidget(SubtitleLabel('视频格式转换'))

        row = QHBoxLayout()
        self.lbl = QLabel('未选择文件')
        self.lbl.setStyleSheet('color:gray;')
        row.addWidget(self.lbl)
        btn_sel = PushButton('选择 MP4 文件')
        btn_sel.clicked.connect(self.pick_file)
        row.addWidget(btn_sel)
        layout.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.btn_go = PrimaryPushButton('转换为 AVI')
        self.btn_go.clicked.connect(self.convert)
        self.btn_go.setEnabled(False)
        layout.addWidget(self.btn_go)

    def pick_file(self):
        f, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'MP4 (*.mp4)')
        if f:
            self.src_file = f
            self.lbl.setText(os.path.basename(f))
            self.btn_go.setEnabled(True)

    def convert(self):
        d = QFileDialog.getExistingDirectory(self, '保存到')
        if not d:
            return
        name = os.path.splitext(os.path.basename(self.src_file))[0]
        dst = os.path.join(d, name + '.avi')

        self.btn_go.setEnabled(False)
        self.bar.setValue(0)

        self.t = ConvertThread(self.src_file, dst)
        self.t.progress.connect(self.bar.setValue)
        self.t.done.connect(lambda p: (
            QMessageBox.information(self, '完成', f'已保存到:\n{p}'),
            self.btn_go.setEnabled(True)
        ))
        self.t.error.connect(lambda m: (
            QMessageBox.warning(self, '失败', m),
            self.btn_go.setEnabled(True)
        ))
        self.t.start()
