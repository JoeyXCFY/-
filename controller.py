import sys
import os
import cv2

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage ,QPixmap
from PyQt5.QtWidgets import QDialog ,QApplication
from PyQt5.uic import loadUi
from UI import Ui_MainWindow
from subUI import Ui_Dialog
from subUI2 import Ui_Dialog as UD2

subLogic = 0
g_count = 0
RLogic = 0

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        # in python3, super(Class, self).xxx = super().xxx
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.img_path= "./image/"
        self.logic= 0
        self.exit= 0
        self.count= 0

        self.ui.text.setText('press "show" to connect with webcam')
        self.ui.show.clicked.connect(self.onClicked)
        self.ui.capture.clicked.connect(self.captureClicked)
        self.ui.exit.clicked.connect(self.exitCamera)
        self.ui.bokeh.clicked.connect(self.bokeh)
    def onClicked(self):
        self.ui.text.setText('press "capture" to capture image')
        global RLogic

        cap = cv2.VideoCapture(0)

        while(cap.isOpened()):
            if(RLogic == 1):
                self.ui.text.setText('press "capture" to capture image')
                RLogic = 0
            
            ret, frame = cap.read()
            if ret == True:
                self.displayImage(frame, 1)
                cv2.waitKey()
                
                if(self.logic == 2):
                    cv2.imwrite(self.img_path+'%d.jpg'%(self.count), frame)
                    self.ui.text.setText('press "bokeh" to bokeh image')
                    self.logic= 1
                if(self.exit == 1):
                    self.exit= 0
                    break

            else:
                print('return not found')
                break
        cap.release()
        cv2.destroyAllWindows()

    def captureClicked(self):
        self.logic= 2
        #self.count= self.count+1

    def exitCamera(self):
        self.exit= 1
        window.close()
    
    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if(img.shape[2]) == 4:
                qformat = QImage.Format_RGBA888
            
            else:
                qformat = QImage.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        img = img.rgbSwapped()
        self.ui.imgLabel.setPixmap(QPixmap.fromImage(img))
        self.ui.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    
    def bokeh(self):
        os.system('python maskrcnn.py')
        os.system('python opencv.py')
        os.system('python bggan.py')
        self.showbokeh()
        
    def showbokeh(self):
        global subLogic
        subLogic = 1
        child = SubWindow()
        child.exec()

class SubWindow(QtWidgets.QDialog):
    def __init__ (self):
        QDialog.__init__(self)
        self.child = Ui_Dialog()
        self.child.setupUi(self)
        self.setup_control()
        print('subwindow open')
    
    def setup_control(self):
        self.child.btn_org.clicked.connect(self.saveOrgPicture)
        self.child.btn_body.clicked.connect(self.saveBodyPicture)
        self.child.btn_face.clicked.connect(self.saveFacePicture)
        self.child.btn_objcet.clicked.connect(self.saveObjectPicture)
        self.body_path = "./maskrcnnResult/"
        self.face_path = "./opencvResult/"
        self.object_path = "./bgganResult/"
        self.org_path = "./image/"
        self.result_path = "./result/"

        self.displayImg()

    def displayImg(self):
        if(subLogic == 1):
            self.img_org = cv2.imread(self.org_path+'0.jpg')
            self.img_body = cv2.imread(self.body_path+'1.jpg')
            if os.path.isfile(self.face_path+'1.jpg'):
                self.img_face = cv2.imread(self.face_path+'1.jpg')
            else:
                self.img_face = cv2.imread(self.org_path+'0.jpg')
            self.img_object = cv2.imread(self.object_path+'0.jpg')
            height= 480
            width= 640
            bytesPerline = 3 * width
            self.qimg_org = QImage(self.img_org, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.child.origin.setPixmap(QPixmap.fromImage(self.qimg_org))
            self.qimg_body = QImage(self.img_body, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.child.body.setPixmap(QPixmap.fromImage(self.qimg_body))
            self.qimg_face = QImage(self.img_face, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.child.face.setPixmap(QPixmap.fromImage(self.qimg_face))
            self.qimg_object = QImage(self.img_object, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.child.object.setPixmap(QPixmap.fromImage(self.qimg_object))
        
    def saveOrgPicture(self):
        cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_org)
        self.delet()

    def saveBodyPicture(self):
        global grades_path
        grades_path=self.body_path
        #cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_body)
        self.grades()
        #self.delet()
        self.close()

    def saveFacePicture(self):
        global grades_path
        if os.path.isfile(self.face_path+'1.jpg'):
            grades_path=self.face_path
            self.grades()
        #cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_face)
        else:
            cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_org)
        #self.delet()
        self.close()
    
    def saveObjectPicture(self):
        cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_object)
        self.delet()

    def grades(self):
        child2 = Sub2()
        child2.exec()

    def delet(self):
        os.remove(self.org_path+'0.jpg')

        for i in range(4):
            os.remove(self.body_path+ '%d.jpg'%(i))
            if os.path.isfile(self.face_path+'%d.jpg'%(i)):
                os.remove(self.face_path+'%d.jpg'%(i))

        os.remove(self.object_path+'0.jpg')
        self.close()   


class Sub2(QtWidgets.QDialog):

    def __init__ (self):
        QDialog.__init__(self)
        self.child2 = UD2()
        self.child2.setupUi(self)
        self.setup_control()
    
    def setup_control(self):
        self.child2.btn_20.clicked.connect(self.save20)
        self.child2.btn_40.clicked.connect(self.save40)
        self.child2.btn_80.clicked.connect(self.save80)
        self.child2.btn_200.clicked.connect(self.save200)
        self.body_path = "./maskrcnnResult/"
        self.face_path = "./opencvResult/"
        self.object_path = "./bgganResult/"
        self.org_path = "./image/"
        self.result_path="./result/"
        
        self.displayImg()

    def displayImg(self):
        global grades_path
        self.img_20 = cv2.imread(grades_path+'0.jpg')
        self.img_40 = cv2.imread(grades_path+'1.jpg')
        self.img_80 = cv2.imread(grades_path+'2.jpg')
        self.img_200 = cv2.imread(grades_path+'3.jpg')

        height= 480
        width= 640
        bytesPerline = 3 * width
        self.qimg_20 = QImage(self.img_20, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.child2.img_20.setPixmap(QPixmap.fromImage(self.qimg_20))
        self.qimg_40 = QImage(self.img_40, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.child2.img_40.setPixmap(QPixmap.fromImage(self.qimg_40))
        self.qimg_face = QImage(self.img_80, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.child2.img_80.setPixmap(QPixmap.fromImage(self.qimg_face))
        self.qimg_object = QImage(self.img_200, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.child2.img_200.setPixmap(QPixmap.fromImage(self.qimg_object))
        

    def save20(self):
        cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_20)
        self.delet()

    def save40(self):
        cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_40)
        self.delet()

    def save80(self):
        cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_80)
        self.delet()

    def save200(self):
        cv2.imwrite(self.result_path+ '%d.jpg'%(g_count), self.img_200)
        self.delet()

    def delet(self):
        os.remove(self.org_path+'0.jpg')

        for i in range(4):
            os.remove(self.body_path+ '%d.jpg'%(i))
            if os.path.isfile(self.face_path+'%d.jpg'%(i)):
                os.remove(self.face_path+'%d.jpg'%(i))

        os.remove(self.object_path+'0.jpg')
        self.exit()   

    def exit(self):
        global g_count
        global RLogic
        g_count= g_count+1
        RLogic = 1
        self.close()
        print("your image saved")



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())