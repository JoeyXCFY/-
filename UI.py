# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camera2.0.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(847, 570)
        MainWindow.setStyleSheet("background-color: rgb(255, 215, 105)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.text = QtWidgets.QLabel(self.centralwidget)
        self.text.setMaximumSize(QtCore.QSize(656, 16777215))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(20)
        self.text.setFont(font)
        self.text.setFocusPolicy(QtCore.Qt.NoFocus)
        self.text.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.text.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.text.setStyleSheet("color: rgb(0, 0, 0)")
        self.text.setAlignment(QtCore.Qt.AlignCenter)
        self.text.setObjectName("text")
        self.verticalLayout_2.addWidget(self.text)
        self.imgLabel = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel.setMinimumSize(QtCore.QSize(656, 402))
        self.imgLabel.setStyleSheet("border-image: url(:/newPrefix/1.jpg)")
        self.imgLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.imgLabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.imgLabel.setLineWidth(5)
        self.imgLabel.setText("")
        self.imgLabel.setObjectName("imgLabel")
        #self.imgLabel.setScaledContents(True)
        self.verticalLayout_2.addWidget(self.imgLabel)
        self.verticalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.setStretch(1, 4)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.exit = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(20)
        self.exit.setFont(font)
        self.exit.setStyleSheet("background-color: rgb(219, 254, 255)")
        self.exit.setObjectName("exit")
        self.verticalLayout.addWidget(self.exit)
        self.show = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(20)
        self.show.setFont(font)
        self.show.setStyleSheet("background-color: rgb(219, 254, 255)")
        self.show.setObjectName("show")
        self.verticalLayout.addWidget(self.show)
        self.capture = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(20)
        self.capture.setFont(font)
        self.capture.setStyleSheet("background-color: rgb(219, 254, 255)")
        self.capture.setObjectName("capture")
        self.verticalLayout.addWidget(self.capture)
        self.bokeh = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(20)
        self.bokeh.setFont(font)
        self.bokeh.setStyleSheet("background-color: rgb(219, 254, 255)")
        self.bokeh.setObjectName("bokeh")
        self.verticalLayout.addWidget(self.bokeh)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.setStretch(0, 4)
        self.horizontalLayout.setStretch(1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 847, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.text.setText(_translate("MainWindow", "press \"show\" to connect with webcam"))
        self.exit.setText(_translate("MainWindow", "exit"))
        self.show.setText(_translate("MainWindow", "show"))
        self.capture.setText(_translate("MainWindow", "capture"))
        self.bokeh.setText(_translate("MainWindow", "bokeh"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
