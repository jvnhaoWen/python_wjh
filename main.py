import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from qfluentwidgets import FluentWindow, NavigationItemPosition, FluentIcon
from widgets.capture_widget import CaptureWidget
from widgets.playback_widget import PlaybackWidget
from widgets.converter_widget import ConverterWidget


class MainWindow(FluentWindow):

    def __init__(self):
        super().__init__()
        self.resize(960, 640)
        self.setWindowTitle('视频流采集与处理系统')

        # 三个功能页面
        self.capture_page = CaptureWidget(self)
        self.playback_page = PlaybackWidget(self)
        self.converter_page = ConverterWidget(self)

        self.addSubInterface(self.capture_page, FluentIcon.CAMERA, '实时采集')
        self.addSubInterface(self.playback_page, FluentIcon.VIDEO, '视频回放')
        self.addSubInterface(self.converter_page, FluentIcon.SYNC, '格式转换')


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
