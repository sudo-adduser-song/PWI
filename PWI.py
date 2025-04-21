
import pydicom
import cv2 as cv
import numpy as np
import time
from easyocr import Reader
import argparse
import sys

# import matplotlib.pyplot as plt

from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget,QInputDialog,QFrame)


from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QProgressBar



reader=Reader(lang_list=['en'])


def remove_sign(input_string):
    last_dot_index = max(input_string.rfind("-"), input_string.rfind("_"),input_string.rfind("."),input_string.rfind(","),input_string.rfind("="),input_string.rfind("'"),input_string.rfind("\""),input_string.rfind("]"),input_string.rfind("["),input_string.rfind("("),input_string.rfind(")"),input_string.rfind("^"),input_string.rfind("*"),input_string.rfind("<"),input_string.rfind(">"),input_string.rfind("~"))
    
    if last_dot_index != -1 and last_dot_index == len(input_string) - 1:
        cleaned_string = input_string[:last_dot_index]
    else:
        cleaned_string = input_string
    
    return cleaned_string

def cal_checker(max_value, mid_value, min_value):
    check_1 = round(max_value-mid_value,0)
    check_2 = round(mid_value-min_value,0)
    check_diff = abs(check_1-check_2)
    if check_diff == 0 or check_diff == 1 :
        return 1
    else : 
        return 0


img_draw=np.zeros((512,512,3))
roi0_img_mask=np.zeros((512,512,3),dtype=np.uint8)
roi1_img_mask=np.zeros((512,512,3),dtype=np.uint8)
tmp_mask=np.zeros((512,512,3),dtype=np.uint8)
tmp_img=np.zeros((512,512,3))



indices=['CBF','CBV','MTT','TTP','T0']
status_text=''
result1_text='\tRoI 1\n\tCBF:\n\tCBV:\n\tMTT:\n\tTTP:\n\tT0:'
result2_text='\tRoI 2\n\tCBF:\n\tCBV:\n\tMTT:\n\tTTP:\n\tT0:'
checked = [False,False,False,False,False,False] 
user_name=''
class WidgetGallery(QDialog):
    def __init__(self, parent=None):
        global user_name
        super(WidgetGallery, self).__init__(parent)
        user_name,_=QInputDialog.getText(self,'Notice','Input the patient number')
        self.originalPalette = QApplication.palette()
        self.setFixedSize(512, 512)  


        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)
        self.patientInfoLabel = QLabel()
        self.result_2 = QLabel()
        self.result_1 = QLabel()
        # self.createProgressBar()
        self.createTopLeftGroupBox()
        self.createTopRightGroupBox()
        self.createBottomLeftGroupBox()
        self.createBottomRightGroupBox()

        
        
        styleComboBox.textActivated.connect(self.changeStyle)
        self.useStylePaletteCheckBox.toggled.connect(self.changePalette)
        

            
        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)

     
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 0, 0, 1, 2)
        mainLayout.addWidget(self.topLeftGroupBox, 1, 0)
        mainLayout.addWidget(self.topRightGroupBox, 1, 1)
        mainLayout.addWidget(self.bottomLeftGroupBox, 2, 0)
        mainLayout.addWidget(self.bottomRightGroupBox, 2, 1)

        mainLayout.setRowStretch(1, 1)
        mainLayout.setRowStretch(2, 1)
        mainLayout.setColumnStretch(0, 1)
        mainLayout.setColumnStretch(1, 1)
        self.setLayout(mainLayout)
        
        
        self.setWindowTitle("PWI")
        self.changeStyle('Windows')
    
    def changeStyle(self, styleName):
        QApplication.setStyle(QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        if (self.useStylePaletteCheckBox.isChecked()):
            QApplication.setPalette(QApplication.style().standardPalette())
        else:
            QApplication.setPalette(self.originalPalette)


    def createTopLeftGroupBox(self):
        self.topLeftGroupBox = QGroupBox("Type")

        checkbox1 = QCheckBox("CBF")
        checkbox2 = QCheckBox("CBV")
        checkbox3 = QCheckBox("MTT")
        checkbox4 = QCheckBox("TTP")
        checkbox5 = QCheckBox("T0")
        checkbox6 = QCheckBox("Calculate Volume mean")
        
        def handleCheckbox(checkbox, index):
            if checkbox.isChecked():
                # print(f"{checkbox.text()} {index} is checked")
                checked[index] = True
            else:
                # print(f"{checkbox.text()} {index} is unchecked")
                checked[index] = False

        checkbox1.stateChanged.connect(lambda: handleCheckbox(checkbox1, 0))
        checkbox2.stateChanged.connect(lambda: handleCheckbox(checkbox2, 1))
        checkbox3.stateChanged.connect(lambda: handleCheckbox(checkbox3, 2))
        checkbox4.stateChanged.connect(lambda: handleCheckbox(checkbox4, 3))
        checkbox5.stateChanged.connect(lambda: handleCheckbox(checkbox5, 4))
        checkbox6.stateChanged.connect(lambda: handleCheckbox(checkbox6, 5))

        layout = QVBoxLayout()
        layout.addWidget(checkbox1)
        layout.addWidget(checkbox2)
        layout.addWidget(checkbox3)
        layout.addWidget(checkbox4)
        layout.addWidget(checkbox5)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)
        layout.addWidget(checkbox6)
        layout.addStretch(1)
        self.topLeftGroupBox.setLayout(layout)
    
    def updatePatientInfo(self, pn, index, frame, state):
        # self.patientInfoText.clear()
        Text = (
                f"Patient Number: {pn}\n\n"
                f"Type: {indices[index]}\n\n"
                f"Frame: {frame}\n\n"
                f"State: {state}\n\n"
        )
        self.patientInfoLabel.setText(Text)


    def createTopRightGroupBox(self):

        self.topRightGroupBox = QGroupBox("Patient's Info")
        layout = QVBoxLayout()
        layout.addWidget(self.patientInfoLabel)
        layout.addStretch(1)
        self.topRightGroupBox.setLayout(layout)
    
    def update_ROI1_info(self,result):
        Text = (
                f"CBF: {result[0]}\n\n"
                f"CBV: {result[1]}\n\n"
                f"MTT: {result[2]}\n\n"
                f"TTP: {result[3]}\n\n"
                f"T0: {result[4]}\n\n"
        )
        self.result_1.setText(Text)

    def createBottomLeftGroupBox(self):
        self.bottomLeftGroupBox = QGroupBox("ROI 1")
        self.bottomLeftTabWidget = QTabWidget()
        self.bottomLeftTabWidget.setSizePolicy(QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Ignored)
        layout = QGridLayout()
        layout.addWidget(self.result_1)
        self.bottomLeftGroupBox.setLayout(layout)
    
    def update_ROI2_info(self,result):
        Text = (
                f"CBF: {result[0]}\n\n"
                f"CBV: {result[1]}\n\n"
                f"MTT: {result[2]}\n\n"
                f"TTP: {result[3]}\n\n"
                f"T0: {result[4]}\n\n"
        )
        self.result_2.setText(Text)
        
    def createBottomRightGroupBox(self):
        self.bottomRightGroupBox = QGroupBox("ROI 2")
        self.bottomRightTabWidget = QTabWidget()
        self.bottomRightTabWidget.setSizePolicy(QSizePolicy.Policy.Preferred,
                QSizePolicy.Policy.Ignored)
        layout = QGridLayout()
        layout.addWidget(self.result_2)
        self.bottomRightGroupBox.setLayout(layout)

    def put_status_text(self,new_text):
        self.status_label.clear()                # 지우기
        self.status_label.setText(new_text)     # 텍스트 출력
        
    def put_result1_text(self,new_text):
        self.result1_label.clear()
        self.result1_label.setText(new_text)
        
    def put_result2_text(self,new_text):
        self.result2_label.clear()
        self.result2_label.setText(new_text)


isDragging = False
x0, y0= -1, -1

tmp_x0,tmp_x1,tmp_y0,tmp_y1=-1,-1,-1,-1
index_name = ['CBF','CBV','MTT','TTP','T0']

roi_mouse_flag=0

def calculate_1D_array(roi_1d_img,dictionary):

    matched_pixel=np.array([dictionary[(b,g,r)] for [b,g,r] in roi_1d_img if (b,g,r) in dictionary])
    unmatched_pixel=np.array([[b,g,r] for [b,g,r] in roi_1d_img if (b,g,r) not in dictionary])
    sum = np.sum(matched_pixel)
    count= matched_pixel.shape[0]
    dict_keys=np.array(list(dictionary.keys()))
    dict_values=np.array(list(dictionary.values()))
    
    for pixel in unmatched_pixel:
        sum+=dict_values[np.argmin(np.linalg.norm(dict_keys-pixel, axis=0),axis=0)]
        count+=1
    
            
    return round(sum/count,2)
      
def process_image_closed_curve(process_index,frame,roi0_info,roi0_img_mask,roi1_info,roi1_img_mask,total_img,results_0,results_1):    
    roi0_count_sum = 0  # 변수 초기화
    roi1_count_sum = 0  # 변수 초기화
    roi0_sum = 0  # 변수 초기화
    roi1_sum = 0  # 변수 초기화
    
    index_img=total_img[:,frame].copy()
    img=index_img[process_index]
    dictionary = {}
    bar_rgb = []
    
    
    for f_num in range(22):
        roi0_img=total_img[process_index,f_num].copy()
        roi1_img=total_img[process_index,f_num].copy()

        roi0_img=roi0_img*roi0_img_mask
        roi1_img=roi1_img*roi1_img_mask
        
        roi0_img=roi0_img[roi0_info[1]:roi0_info[1]+roi0_info[3],roi0_info[0]:roi0_info[0]+roi0_info[2]]
        roi1_img=roi1_img[roi1_info[1]:roi1_info[1]+roi1_info[3],roi1_info[0]:roi1_info[0]+roi1_info[2]]
        dictionary,bar_rgb = bar_dic_calculator(img)
        
        #ROI_0 weighted
        count0 = 0 
        sum0 = 0
        b_color = img[3,3]
        count0, sum0 = weight_calculator(roi0_img,dictionary,bar_rgb,img,b_color)
        roi0_count_sum += count0
        roi0_sum +=sum0
        
        #ROI_1 weighted
        count1=0
        sum1 = 0
        count1, sum1 = weight_calculator(roi1_img,dictionary,bar_rgb,img,b_color)
        roi1_count_sum += count1
        roi1_sum +=sum1

        if roi1_count_sum != 0:
            roi1_sum_avg = round(roi1_sum / roi1_count_sum, 4)
        else:
            roi1_sum_avg = 0.0  # 또는 다른 기본값으로 설정할 수 있음

        if roi0_count_sum != 0:
            roi0_sum_avg = round(roi0_sum / roi0_count_sum, 4)
        else:
            roi0_sum_avg = 0.0  # 또는 다른 기본값으로 설정할 수 있음
    # print(round(roi0_sum_avg,2))
    # print(round(roi1_sum_avg,2))
    results_0[process_index]= round(roi0_sum_avg,2)
    results_1[process_index]= round(roi1_sum_avg,2)

def preprocess(roi_img_mask):
    roi_img_mask=cv.dilate(roi_img_mask,np.ones((3,3),dtype=np.uint8),iterations=1)
    roi_img_mask_gray=cv.cvtColor(roi_img_mask,cv.COLOR_BGR2GRAY)
    roi_x,roi_y,roi_w,roi_h=cv.boundingRect(roi_img_mask_gray)
    ret, thresh = cv.threshold(roi_img_mask_gray, 0, 255, 0)
    contours, hierarchy = cv.findContours(thresh, 1,2)
    roi_info=[]
    roi_info.append(roi_x)
    roi_info.append(roi_y)
    roi_info.append(((roi_w+1)//2)*2)
    roi_info.append(((roi_h+1)//2)*2)
    floodfill(roi_img_mask[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w],roi_w,roi_h,contours[0][0,0,0]-roi_x,contours[0][0,0,1]-roi_y)
    return roi_img_mask, roi_info


def isValid(img, w, h, x, y):
    if x<0 or x>= w or y<0 or y>= h:
        return False
    elif (img[y,x]==[1,1,1]).all():
        return False
    return True

def floodfill(img, w, h, x, y):
    queue = []
     
    # Append the position of starting 
    # pixel of the component
    queue.append([x, y])
 
    # While the queue is not empty i.e. the 
    # whole component having prevC color 
    # is not colored with newC color
    while queue:
        
        # Dequeue the front node
        currPixel = queue.pop()
         
        posX = currPixel[0]
        posY = currPixel[1]
         
        # Check if the adjacent
        # pixels are valid
        if isValid(img, w, h, posX + 1, posY):
            img[posY,posX + 1] = [1,1,1]
            queue.append([posX + 1, posY])
         
        if isValid(img, w, h, posX-1, posY):
            img[posY,posX-1]= [1,1,1]
            queue.append([posX-1, posY])
         
        if isValid(img, w, h, posX, posY + 1):
            img[posY + 1,posX]= [1,1,1]
            queue.append([posX, posY + 1])
         
        if isValid(img, w, h, posX, posY-1):
            img[posY-1,posX]= [1,1,1]
            queue.append([posX, posY-1])


which_color = 0

def onMouse(event, x1, y1, flags, param):
    global isDragging, x0, y0
    global tmp_x0,tmp_x1,tmp_y0,tmp_y1
    global roi_mouse_flag
    global lr_click_flag 
    global tmp_roi1_info, tmp_roi0_info,roi1_info, roi0_info

    img =param
    
    if which_color == 0 :
        color = (255,0,0)
    else : 
        color = (0,0,255)
    if event == cv.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x1
        y0 = y1
        tmp_x0=x1
        tmp_y0=y1
        
        
    elif event == cv.EVENT_MOUSEMOVE :
        if isDragging:
            img_draw = img.copy()
            cv.ellipse(img_draw, (((x1+x0)/2, (y1+y0)/2), ((x1-x0), (y1-y0)),0),color, 1)
            
            cv.imshow('img', img_draw)
    elif event == cv.EVENT_LBUTTONUP :
        if isDragging:
            isDragging = False
            w0 = x1 - x0
            h0 = y1 - y0
            if w0 > 15 and h0 > 15:
                tmp_x1=x1
                tmp_y1=y1
                img_draw = img.copy()
                cv.ellipse(img_draw, (((x1+x0)/2, (y1+y0)/2), ((x1-x0), (y1-y0)),0),color, 1)
                roi_mouse_flag = 1 
                cv.imshow('img', img_draw)
            else:
                print('drag should start from left-top side')
   

def onMouse_clicked(event, x1, y1, flags, param):
    img =param

    global isDragging, x0, y0
    global tmp_x0,tmp_x1,tmp_y0,tmp_y1
    global roi_mouse_flag

    global isDragging, x0, y0, tmp_x0, tmp_x1, tmp_y0, tmp_y1, roi_mouse_flag, tmp_roi1_info, tmp_roi0_info,roi1_info, roi0_info

    if event == cv.EVENT_RBUTTONDOWN:
        isDragging = True
        x0 = x1
        y0 = y1
        tmp_x0=x1
        tmp_y0=y1

    if event == cv.EVENT_RBUTTONUP :
        if tmp_roi0_info:
            center_x, center_y, width, height = tmp_roi0_info
            if x0+width//2 >512 or y0+height//2 >512 :
                print("Draw Again")
                roi_mouse_flag = 0
                cv.imshow('img',img.copy())
            else :
                img_draw = img.copy()
                cv.ellipse(img_draw, (x1, y1), (width//2, height//2), 0, 0, 360, (0, 0, 255), 1)
                roi_mouse_flag = 1
                cv.imshow('img', img_draw)
                isDragging=False

roi_mask_symmetric=np.zeros((512,512,3),dtype=np.uint8)
tmp_roi_mask_symmetric=np.zeros((512,512,3),dtype=np.uint8)

def onMouse_closed_curve(event, x1, y1, flags, param):
    global isDragging, x0, y0
    global tmp_x0,tmp_x1,tmp_y0,tmp_y1
    global img_draw,tmp_img,tmp_mask
    global roi0_img_mask,roi1_img_mask,roi_mouse_flag, tmp_roi_mask_symmetric
    img =param
    color=(0,0,0)
    if which_color == 0:
        color=(0,0,255)
        img_draw=tmp_img
        roi0_img_mask=tmp_mask
    elif which_color == 1:
        color=(255,0,0)
        img=img_draw
        roi1_img_mask=tmp_mask
        
        
    if event == cv.EVENT_LBUTTONDOWN:
        tmp_img = img.copy()
        tmp_mask=np.zeros((512,512,3),dtype=np.uint8)
        isDragging = True
        x0 = x1
        y0 = y1
        tmp_x0=x1
        tmp_y0=y1
    elif event == cv.EVENT_MOUSEMOVE:
        if isDragging:
            cv.line(tmp_img,(tmp_x0,tmp_y0),(x1,y1),color,2)
            cv.line(tmp_mask,(tmp_x0,tmp_y0),(x1,y1),(1,1,1),2)
            tmp_x0=x1
            tmp_y0=y1
            cv.imshow('img', tmp_img)
    elif event == cv.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            tmp_mask_check=cv.dilate(tmp_mask,np.ones((3,3),np.uint8),iterations=1)
            contour_check=cv.cvtColor(tmp_mask_check,cv.COLOR_BGR2GRAY)
            contours_t, hierarchy = cv.findContours(contour_check, 1,2)
            if len(contours_t)>2:
                print('Draw again!')
                roi_mouse_flag = 0
                cv.imshow('img',img.copy())
            elif np.linalg.norm(np.array([x0,y0])-np.array([x1,y1]))>=20:
                print('Draw again!')
                roi_mouse_flag = 0
                cv.imshow('img', img.copy())
            else:
                cv.line(tmp_img,(x0,y0),(x1,y1),color,2)
                cv.line(tmp_mask,(x0,y0),(x1,y1),(1,1,1),2)
                roi_mouse_flag = 1 
                tmp_roi_mask_symmetric=tmp_mask[:,::-1,:]
                cv.imshow('img', tmp_img)


def onMouse_closed_curve_clicked(event, x1, y1, flags, param):
    img =param

    global isDragging, x0, y0, img_draw
    global tmp_x0,tmp_x1,tmp_y0,tmp_y1
    global roi_mouse_flag

    global isDragging, x0, y0, tmp_x0, tmp_x1, tmp_y0, tmp_y1, roi_mouse_flag, tmp_roi1_info, tmp_roi0_info,roi1_info, roi0_info,roi1_img_mask

    if event == cv.EVENT_RBUTTONDOWN:
        isDragging = True
        x0 = x1
        y0 = y1
        tmp_x0=x1
        tmp_y0=y1

    elif event == cv.EVENT_RBUTTONUP :
        # if tmp_roi0_info:
        # center_x, center_y, width, height = tmp_roi0_info
        tmp_img_draw = img_draw.copy()
        if(x1-tmp_roi0_info[2]//2<0 or x1+tmp_roi0_info[2]//2>511 or y1-tmp_roi0_info[3]//2<0 or y1+tmp_roi0_info[3]//2>511):
            print('Draw Again!')
        else:
            tmp_img_draw[y1-tmp_roi0_info[3]//2:y1+tmp_roi0_info[3]//2,x1-tmp_roi0_info[2]//2:x1+tmp_roi0_info[2]//2]=tmp_img_draw[y1-tmp_roi0_info[3]//2:y1+tmp_roi0_info[3]//2,x1-tmp_roi0_info[2]//2:x1+tmp_roi0_info[2]//2]*(1-tmp_roi_mask_symmetric[tmp_roi0_info[1]:tmp_roi0_info[1]+tmp_roi0_info[3],511-tmp_roi0_info[0]-tmp_roi0_info[2]:511-tmp_roi0_info[0],:])
            tmp_img_draw[y1-tmp_roi0_info[3]//2:y1+tmp_roi0_info[3]//2,x1-tmp_roi0_info[2]//2:x1+tmp_roi0_info[2]//2]=tmp_img_draw[y1-tmp_roi0_info[3]//2:y1+tmp_roi0_info[3]//2,x1-tmp_roi0_info[2]//2:x1+tmp_roi0_info[2]//2]+(np.multiply(tmp_roi_mask_symmetric[tmp_roi0_info[1]:tmp_roi0_info[1]+tmp_roi0_info[3],511-tmp_roi0_info[0]-tmp_roi0_info[2]:511-tmp_roi0_info[0],:],[255,0,0]))
            # tmp_roi1_info = [x1-width//2, y1-height//2, width, height]
            # roi1_info = tmp_roi1_info.copy()
            symmetric_mask=np.zeros((512,512,3),dtype=np.uint8)
            symmetric_mask[y1-tmp_roi0_info[3]//2:y1+tmp_roi0_info[3]//2,x1-tmp_roi0_info[2]//2:x1+tmp_roi0_info[2]//2]=symmetric_mask[y1-tmp_roi0_info[3]//2:y1+tmp_roi0_info[3]//2,x1-tmp_roi0_info[2]//2:x1+tmp_roi0_info[2]//2]+tmp_roi_mask_symmetric[tmp_roi0_info[1]:tmp_roi0_info[1]+tmp_roi0_info[3],511-tmp_roi0_info[0]-tmp_roi0_info[2]:511-tmp_roi0_info[0],:]
            
            roi1_img_mask=symmetric_mask.copy()
            roi_mouse_flag = 1
            isDragging=False
            cv.imshow('img', tmp_img_draw)

def cal_ellip_locate (tmp_x1,tmp_x0,tmp_y0,tmp_y1):
    return (tmp_x1 + tmp_x0) // 2, (tmp_y1 + tmp_y0) // 2,tmp_x1 - tmp_x0, tmp_y1 - tmp_y0
    


def bar_dic_calculator(img):
    tmp_gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    ret,thresh = cv.threshold(tmp_gray,150,255,0)
    contours, hierarchy = cv.findContours(thresh, 1,2)

    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True), True)
        if len(approx) == 4:
            xx, yy, ww, hh = cv.boundingRect(cnt)
            if hh>100:
                x,y,w,h=xx,yy,ww,hh
    midth=int(x+w/2)
    dictionary={}
    bar_rgb=[]

    for i in range(h-3):
        dictionary[tuple(img[y+1+i,midth])]=0
        bar_rgb.append(img[y+1+i,midth])

    moved_x = 50 
    bar_roi = img[0:512,x-moved_x:512]

    height, width = bar_roi.shape[0], bar_roi.shape[1]
    text_drawed = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height): 
        for j in range(width):
            if tuple(bar_roi[i,j]) in dictionary:
                continue
            else :
                tmp1=np.int16(bar_roi[i,j])
                f_min=255*3
                f_near=bar_roi[i,j]
                f_gap=np.int16(0)
                for key in bar_rgb:
                    tmp2=np.int16(key)
                    f_gap=abs(tmp1[0]-tmp2[0])+abs(tmp1[1]-tmp2[1])+abs(tmp1[2]-tmp2[2])
                    if f_gap < f_min: 
                        f_min= f_gap
                if f_min >= 10 :
                    text_drawed[i,j]= bar_roi[i,j]

    cv.rectangle(text_drawed, (moved_x-10,y-2), (width,y+h+2), (0,0,0), -1)
    g_text_drawed=cv.cvtColor(text_drawed,cv.COLOR_RGB2GRAY)

    top_roi = cv.resize(g_text_drawed[y-50:y+20,0:x+30],(100,100),cv.INTER_CUBIC)
    center_roi = cv.resize(g_text_drawed[y+h//2-30:y+h//2+30,0:x+30],(100,100),cv.INTER_CUBIC)
    bottom_roi = cv.resize(g_text_drawed[y+h-30:y+h+20,0:x+30],(100,100),cv.INTER_CUBIC)
    
    draft_text = []
    sec_draft_text = []
    result = []

    top_text = reader.readtext(top_roi, detail=0) 
    center_text = reader.readtext(center_roi, detail=0)
    bottom_text = reader.readtext(bottom_roi, detail=0) 
    
    for i in range(len(top_text)):  
            draft_text.append(top_text[i])  
    for i in range(len(center_text)):
            draft_text.append(center_text[i])  
    for i in range(len(bottom_text)):
            draft_text.append(bottom_text[i])
    
    for i in range(len(draft_text)):
        if len(draft_text[i])>0:
            sec_draft_text.append(remove_sign(draft_text[i]))

    sec_draft_text = sec_draft_text[1:]
    result = []

    for i in range(len(sec_draft_text)):
        try :
            result.append(float(sec_draft_text[i]))
        except:
            continue
    
    if(len(result)<3):
            print("(Error) Wrong Text ")
            print("Please manually input the max, mid, and min values.")
            max_value = float(input("Enter max_value: "))
            mid_value = float(input("Enter mid_value : "))
            min_value = float(input("Enter min_value : "))
    
    max_value = result[0]
    mid_value = result[1]
    min_value = result[2]

    check = cal_checker(max_value,mid_value,min_value)
    if check == 0 :  
        while cal_checker(max_value,mid_value,min_value)!=1:
            print("(Error) Wrong Text ")
            print("Please manually input the max, mid, and min values.")
            max_value = float(input("Enter max_value: "))
            mid_value = float(input("Enter mid_value : "))
            min_value = float(input("Enter min_value : "))
    
    step_weight =  (max_value - min_value) / (h-3)
    
    dictionary={}
    bar_rgb=[]
    
    for i in range(h-3):
        dictionary[tuple(img[y+1+i,midth])]=max_value-(step_weight*i)
        bar_rgb.append(img[y+1+i,midth])
    
    return dictionary,bar_rgb


def weight_calculator(ellipsis_roi, dictionary, bar_rgb, img,b_color):
    count = 0
    sum_val = 0.0
    non_zero_pixels = [[i, j] for i in range(ellipsis_roi.shape[0]) for j in range(ellipsis_roi.shape[1]) if not (ellipsis_roi[i, j] == b_color).all()]
    count = len(non_zero_pixels)
    for [i, j] in non_zero_pixels:
        pixel = ellipsis_roi[i, j]
        if tuple(pixel) in dictionary:
            sum_val += round(dictionary[tuple(pixel)], 2)
            count += 1
        else:
            brain = np.int16(pixel)
            f_min = 255 * 3
            f_near = img[i, j]
            for key in bar_rgb:
                tmp2 = np.int16(key)
                f_gap = abs(brain[0] - tmp2[0]) + abs(brain[1] - tmp2[1]) + abs(brain[2] - tmp2[2])
                if f_gap < f_min:
                    f_near = key
                    f_min = f_gap
            if f_min < 10:
                sum_val += round(dictionary[tuple(f_near)], 2)
                count += 1
    return count, sum_val

                
    
def find_nearest_rgb(bar_rgb, brain):
    f_min = 255 * 3
    f_near = brain
    for key in bar_rgb:
        tmp2 = np.int16(key)
        f_gap = abs(brain[0] - tmp2[0]) + abs(brain[1] - tmp2[1]) + abs(brain[2] - tmp2[2])
        if f_gap < f_min:
            f_near = key
            f_min = f_gap
    return f_near
  
ttpye  = ['CBF','CBV', 'MTT','TTP','T0']

def process_image(process_index,frame,roi0_info,roi1_info,total_img,results_0,results_1):
    roi0_count_sum = 0  # 변수 초기화
    roi1_count_sum = 0  # 변수 초기화
    roi0_sum = 0  # 변수 초기화
    roi1_sum = 0  # 변수 초기화
    
    index_img=total_img[:,frame].copy()
    img=index_img[process_index]
    dictionary = {}
    bar_rgb = []
    roi0_info.copy()
    roi1_info.copy()

    for f_num in range(22):
        # print("+",end='')
        roi_img=total_img[process_index,f_num].copy()
        roi0_img=total_img[process_index,f_num].copy()
        roi1_img=total_img[process_index,f_num].copy()

        cv.ellipse(roi0_img, ((roi0_info[0],roi0_info[1]), (abs(roi0_info[2]), abs(roi0_info[3])),0),(0,0,0), -1)
        roi0_img=roi_img-roi0_img
        ellipsis_roi0 = roi0_img[roi0_info[1]-roi0_info[3]//2:roi0_info[1]+roi0_info[3]//2,roi0_info[0]-roi0_info[2]//2:roi0_info[0]+roi0_info[2]//2]            

        cv.ellipse(roi1_img, ((roi1_info[0],roi1_info[1]), (abs(roi1_info[2]), abs(roi1_info[3])),0),(0,0,0), -1)
        roi1_img=roi_img-roi1_img
        ellipsis_roi1 = roi1_img[roi1_info[1]-abs(roi1_info[3])//2:roi1_info[1]+abs(roi1_info[3])//2,roi1_info[0]-abs(roi1_info[2])//2:roi1_info[0]+abs(roi1_info[2])//2]
        dictionary,bar_rgb = bar_dic_calculator(img)
        b_color = img[3,3]
        #ROI_0 weighted
        count0 = 0 
        sum0 = 0
        count0, sum0 = weight_calculator(ellipsis_roi0,dictionary,bar_rgb,img,b_color)
        roi0_count_sum += count0
        roi0_sum +=sum0
        
        #ROI_1 weighted
        count1=0
        sum1 = 0
        count1, sum1 = weight_calculator(ellipsis_roi1,dictionary,bar_rgb,img,b_color)
        roi1_count_sum += count1
        roi1_sum +=sum1

        if roi1_count_sum != 0:
            roi1_sum_avg = round(roi1_sum / roi1_count_sum, 4)
        else:
            roi1_sum_avg = 0.0  # 또는 다른 기본값으로 설정할 수 있음

        if roi0_count_sum != 0:
            roi0_sum_avg = round(roi0_sum / roi0_count_sum, 4)
        else:
            roi0_sum_avg = 0.0  # 또는 다른 기본값으로 설정할 수 있음
        # print("+",end='')
    # print(round(roi0_sum_avg,2))
    # print(round(roi1_sum_avg,2))
    results_0[process_index]= round(roi0_sum_avg,2)
    results_1[process_index]= round(roi1_sum_avg,2)
    
    end_2 = time.time()




def file_load(text):
    data_path='./data/'

    user_path=text

    CBF_source_path='/CBF/CBF'
    CBV_source_path='/CBV/CBV'
    MTT_source_path='/MTT/MTT'
    TTP_source_path='/TTP/TTP'
    T0_source_path='/T0/T0'


    total_img=np.empty((1,22,512,512,3),dtype='uint8')

    CBF_img=np.empty((1,512,512,3),dtype='uint8')
    CBV_img=np.empty((1,512,512,3),dtype='uint8')
    MTT_img=np.empty((1,512,512,3),dtype='uint8')
    TTP_img=np.empty((1,512,512,3),dtype='uint8')
    T0_img=np.empty((1,512,512,3),dtype='uint8')
    for i in range(22):
        CBF_path=data_path+user_path+CBF_source_path+'{0:0>4}.dcm'.format(i)
        CBV_path=data_path+user_path+CBV_source_path+'{0:0>4}.dcm'.format(i)
        MTT_path=data_path+user_path+MTT_source_path+'{0:0>4}.dcm'.format(i)
        T0_path=data_path+user_path+T0_source_path+'{0:0>4}.dcm'.format(i)
        TTP_path=data_path+user_path+TTP_source_path+'{0:0>4}.dcm'.format(i)
        
        CBF=(pydicom.dcmread(CBF_path)).pixel_array
        CBV=(pydicom.dcmread(CBV_path)).pixel_array
        MTT=(pydicom.dcmread(MTT_path)).pixel_array
        T0=(pydicom.dcmread(T0_path)).pixel_array
        TTP=(pydicom.dcmread(TTP_path)).pixel_array
        
        CBF=cv.cvtColor(CBF,cv.COLOR_BGR2RGB)
        CBV=cv.cvtColor(CBV,cv.COLOR_BGR2RGB)
        MTT=cv.cvtColor(MTT,cv.COLOR_BGR2RGB)
        T0=cv.cvtColor(T0,cv.COLOR_BGR2RGB)
        TTP=cv.cvtColor(TTP,cv.COLOR_BGR2RGB)
        
        CBF_img=np.append(CBF_img,CBF.reshape(1,512,512,3),axis=0)
        CBV_img=np.append(CBV_img,CBV.reshape(1,512,512,3),axis=0)
        MTT_img=np.append(MTT_img,MTT.reshape(1,512,512,3),axis=0)
        T0_img=np.append(T0_img,T0.reshape(1,512,512,3),axis=0)
        TTP_img=np.append(TTP_img,TTP.reshape(1,512,512,3),axis=0)
        
    CBF_img=np.delete(CBF_img,[0,0,0,0],axis=0)
    CBV_img=np.delete(CBV_img,[0,0,0,0],axis=0)
    MTT_img=np.delete(MTT_img,[0,0,0,0],axis=0)
    TTP_img=np.delete(TTP_img,[0,0,0,0],axis=0)
    T0_img=np.delete(T0_img,[0,0,0,0],axis=0)

    total_img=np.append(total_img,CBF_img.reshape(1,22,512,512,3),axis=0)
    total_img=np.append(total_img,CBV_img.reshape(1,22,512,512,3),axis=0)
    total_img=np.append(total_img,MTT_img.reshape(1,22,512,512,3),axis=0)
    total_img=np.append(total_img,TTP_img.reshape(1,22,512,512,3),axis=0)
    total_img=np.append(total_img,T0_img.reshape(1,22,512,512,3),axis=0)

    total_img=np.delete(total_img,[0,0,0,0,0],axis=0)
    
    return total_img


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    total_img = file_load(user_name)
    results_0 = [0] * 5
    results_1 = [0] * 5
    index=0
    frame=0
    
    while(1):
        show_img=total_img[index,frame]
        cv.imshow('img',show_img)
        state = 'Select an Image'    
        gallery.updatePatientInfo(user_name, index, frame,state)
        
        x0, y0= -1, -1
        tmp_x0,tmp_x1,tmp_y0,tmp_y1=-1,-1,-1,-1

        input_c=cv.waitKey()
        
        if input_c ==27: # esc = break
            break
        
        if input_c == 2: # <- = previous frame
            frame=frame-1
        elif input_c == 3: # -> = next frame
            frame=frame+1
        if frame<0: # processing over frame
            frame=21
        elif frame>21: # processing over frame
            frame=0
           
        if input_c == 0 : # up = previous index
            index= index+1
        elif input_c == 1 : # down = next index
            index= index-1 
        if index <0 : # processing over index
            index=4
        elif index >4 : # processing over index
            index=0
       
            
        if input_c == 49: # Ellipse
            roi_img=total_img[index,frame].copy()
            roi0_img=total_img[index,frame].copy()
            roi1_img=total_img[index,frame].copy()
            state = "Draw ellipse ROI 1"
            gallery.updatePatientInfo(user_name, index, frame,state)
            which_color = 0
            cv.imshow('img',roi_img)
            roi_mouse_flag = 0
            roi0_img_mask=np.zeros((512,512,3),dtype=np.uint8)
            roi1_img_mask=np.zeros((512,512,3),dtype=np.uint8)
            
            while(1):   
                cv.setMouseCallback('img', onMouse, param=roi_img)
                k_1_tmp = cv.waitKey(0) 
                if k_1_tmp == 68 or  k_1_tmp == 98 or k_1_tmp == 50:
                    break
            
                if k_1_tmp == 32 and roi_mouse_flag == 1: # space
                    cv.ellipse(roi_img, (((tmp_x1+tmp_x0)/2, (tmp_y1+tmp_y0)/2), (abs(tmp_x1-tmp_x0), abs(tmp_y1-tmp_y0)),0),(255,0,0), 1)
                    cv.ellipse(roi0_img_mask, (((tmp_x1+tmp_x0)/2, (tmp_y1+tmp_y0)/2), (abs(tmp_x1-tmp_x0), abs(tmp_y1-tmp_y0)),0),(1,1,1), -1)

                    center_x_0 , center_y_0 ,e_width_0, e_height_0  = cal_ellip_locate (tmp_x1,tmp_x0,tmp_y0,tmp_y1)
                    roi0_info = []
                    roi0_info.append(center_x_0-e_width_0/2)
                    roi0_info.append(center_y_0-e_height_0/2)
                    roi0_info.append(e_width_0)
                    roi0_info.append(e_height_0)

                    break
            if k_1_tmp == 68 or  k_1_tmp == 98 or k_1_tmp == 50:
                cv.destroyWindow('img')
                continue

            which_color = 1

            state = "Draw ellipse ROI 2"
            tmp_roi0_info = roi0_info.copy()
            gallery.updatePatientInfo(user_name, index, frame,state)
            while(1):
                cv.setMouseCallback('img', onMouse_clicked, param=roi_img)
                k__2_tmp = cv.waitKey(0)
                if k__2_tmp == 68 or  k__2_tmp == 98:
                    break
                
                if k__2_tmp == 32  and roi_mouse_flag == 1:
                    cv.ellipse(roi_img, (tmp_x0, tmp_y0), (roi0_info[2]//2, roi0_info[3]//2),0, 0, 360, (0, 0,255), 1)
                    cv.ellipse(roi1_img_mask, (tmp_x0, tmp_y0), (roi0_info[2]//2, roi0_info[3]//2),0, 0, 360, (1, 1, 1), -1)
                    roi1_info = []
                    roi1_info.append(tmp_x0-roi0_info[2]//2)
                    roi1_info.append(tmp_y0-roi0_info[3]//2)
                    roi1_info.append(roi0_info[2]+1)
                    roi1_info.append(roi0_info[3]+1)
                    lr_click_flag = 0
                    break
                        
            if k__2_tmp == 68 or  k__2_tmp == 98 or k__2_tmp == 50:
                    cv.destroyWindow('img')
                    continue
            
            state = "Calculating..."
            gallery.updatePatientInfo(user_name, index, frame,state)
            
            # print(total_img.shape)
            # print(roi1_img_mask.shape)

            if(checked[5]==False):
                np.floor

            
                selected_frame_img = total_img[:, frame, :, :, :]
                roi0_total_img = np.multiply(selected_frame_img, roi0_img_mask)[:, int(roi0_info[1]-1):int(roi0_info[1])+roi0_info[3]+1, int(roi0_info[0]-1):int(roi0_info[0])+roi0_info[2]+1, :]
                roi1_total_img = np.multiply(selected_frame_img, roi1_img_mask)[:,int(roi1_info[1]-1):int(roi1_info[1])+roi1_info[3]+1,int(roi1_info[0]-1):int(roi1_info[0])+roi1_info[2]+1,:]
                
                roi0_1D_array=roi0_total_img.reshape(5,-1,3)
                roi1_1D_array=roi1_total_img.reshape(5,-1,3)
                
                # tmp=[pixel for pixel in roi0_1D_array[:] if np.sum(pixel)!= 0 ]
                
                # print(roi0_total_img.shape)
                # print(roi1_total_img.shape)
                # print(roi0_1D_array.shape)
                # print(roi1_1D_array.shape)
                
                
                print("Calculating")
                start_1 = time.time()
                
                for type in range(5):
                    if(checked[type]==False):
                        results_0[type]=0
                        results_1[type]=0
                        continue
                    else :
                        dict,_=bar_dic_calculator(total_img[type,21])
                        t1=time.time()
                        results_0[type]=calculate_1D_array(roi0_1D_array[type],dict)
                        results_1[type]=calculate_1D_array(roi1_1D_array[type],dict)
                        t3=time.time()
                        print(f'{indices[type]} finished! Execution Time : {round(t3-t1,2)}')
                        
                # print(results_0)
                # print(results_1)
            
                end = time.time()
                print("Total Execution Time "+str(round(end- start_1,3)))
                state = "Done... Please enter any key"
                gallery.updatePatientInfo(user_name, index, frame,state)
                gallery.update_ROI1_info(results_0)
                gallery.update_ROI2_info(results_1)
                cv.waitKey(0)
                cv.destroyWindow('img')
                which_color = 0
                
                cv.destroyAllWindows()
                
            elif checked[5]==True :
                np.floor
                
                roi0_total_img=np.multiply(total_img,roi0_img_mask)[:,:,int(roi0_info[1]-1):int(roi0_info[1])+roi0_info[3]+1,int(roi0_info[0]-1):int(roi0_info[0])+roi0_info[2]+1,:]
                roi1_total_img=np.multiply(total_img,roi1_img_mask)[:,:,int(roi1_info[1]-1):int(roi1_info[1])+roi1_info[3]+1,int(roi1_info[0]-1):int(roi1_info[0])+roi1_info[2]+1,:]
                
                roi0_1D_array=roi0_total_img.reshape(5,-1,3)
                roi1_1D_array=roi1_total_img.reshape(5,-1,3)
                
                # tmp=[pixel for pixel in roi0_1D_array[:] if np.sum(pixel)!= 0 ]
                
                # print(roi0_total_img.shape)
                # print(roi1_total_img.shape)
                # print(roi0_1D_array.shape)
                # print(roi1_1D_array.shape)
                
                
                print("Calculating")
                start_1 = time.time()
                
                for type in range(5):
                    if(checked[type]==False):
                        results_0[type]=0
                        results_1[type]=0
                        continue
                    else :
                        dict,_=bar_dic_calculator(total_img[type,21])
                        t1=time.time()
                        results_0[type]=calculate_1D_array(roi0_1D_array[type],dict)
                        results_1[type]=calculate_1D_array(roi1_1D_array[type],dict)
                        t3=time.time()
                        print(f'{indices[type]} finished! Execution Time : {round(t3-t1,2)}')
                        
                # print(results_0)
                # print(results_1)
            
                end = time.time()
                print("Total Execution Time "+str(round(end- start_1,3)))
                state = "Done... Please enter any key"
                gallery.updatePatientInfo(user_name, index, frame,state)
                gallery.update_ROI1_info(results_0)
                gallery.update_ROI2_info(results_1)
                cv.waitKey(0)
                cv.destroyWindow('img')
                which_color = 0
                
                cv.destroyAllWindows()



        if input_c == 50:
            roi_img=total_img[index,frame].copy()

            state = "Draw closed curve ROI"
            gallery.updatePatientInfo(user_name, index, frame,state)
        
            cv.imshow('img',roi_img)
            which_color=0
            while(1):
                cv.setMouseCallback('img', onMouse_closed_curve, param=roi_img)
                k_1_tmp= cv.waitKey(0)
                if k_1_tmp == 68 or  k_1_tmp == 98 or k_1_tmp == 50:
                    break
            
                if k_1_tmp == 32 and roi_mouse_flag == 1: # space
                    roi0_img_mask,roi0_info=preprocess(roi0_img_mask)
                    roi_mouse_flag = 0
                    break
            if k_1_tmp == 68 or  k_1_tmp == 98 or k_1_tmp == 50:
                cv.destroyWindow('img')
                continue
            
            tmp_roi0_info = roi0_info.copy()
            
            which_color=1
            while(1):
                cv.setMouseCallback('img', onMouse_closed_curve_clicked, param=roi_img)
                k_2_tmp= cv.waitKey(0)
                if k_2_tmp == 68 or  k_2_tmp == 98 or k_2_tmp == 50:
                    break
            
                if k_2_tmp == 32 and roi_mouse_flag == 1: # space
                    roi1_img_mask,roi1_info=preprocess(roi1_img_mask)
                    roi_mouse_flag = 0
                    break
            if k_1_tmp == 68 or  k_1_tmp == 98 or k_1_tmp == 50:
                cv.destroyWindow('img')
                continue
            
            state = "Calculating..."
            gallery.updatePatientInfo(user_name, index, frame,state)
            
            
            if(checked[5]==True):
                roi0_total_img=np.multiply(total_img,roi0_img_mask)[:,:,roi0_info[1]:roi0_info[1]+roi0_info[3],roi0_info[0]:roi0_info[0]+roi0_info[2],:]
                roi1_total_img=np.multiply(total_img,roi1_img_mask)[:,:,roi1_info[1]:roi1_info[1]+roi1_info[3],roi1_info[0]:roi1_info[0]+roi1_info[2],:]
            else:
                roi0_total_img=np.multiply(total_img,roi0_img_mask)[:,frame,roi0_info[1]:roi0_info[1]+roi0_info[3],roi0_info[0]:roi0_info[0]+roi0_info[2],:]
                roi1_total_img=np.multiply(total_img,roi1_img_mask)[:,frame,roi1_info[1]:roi1_info[1]+roi1_info[3],roi1_info[0]:roi1_info[0]+roi1_info[2],:]
                
            roi0_1D_array=roi0_total_img.reshape(5,-1,3)
            roi1_1D_array=roi1_total_img.reshape(5,-1,3)
            
            # tmp=[pixel for pixel in roi0_1D_array[:] if np.sum(pixel)!= 0 ]
            
            # print(roi0_total_img.shape)
            # print(roi1_total_img.shape)
            # print(roi0_1D_array.shape)
            # print(roi1_1D_array.shape)
            

            print("Calculating")
            start_1 = time.time()
            
            for type in range(5):
                if(checked[type]==False):
                    results_0[type]=0
                    results_1[type]=0
                    continue
                else :
                    dict,_=bar_dic_calculator(total_img[type,21])
                    t1=time.time()
                    results_0[type]=calculate_1D_array(roi0_1D_array[type],dict)
                    results_1[type]=calculate_1D_array(roi1_1D_array[type],dict)
                    t3=time.time()
                    print(f'{indices[type]} finished! Execution Time : {round(t3-t1,2)}')
                    
            # print(results_0)
            # print(results_1)
           
                
            end = time.time()
            print("Total Execution Time "+str(round(end- start_1,3)))
            state = "Done... Please enter any key"
            gallery.updatePatientInfo(user_name, index, frame,state)
            gallery.update_ROI1_info(results_0)
            gallery.update_ROI2_info(results_1)
            cv.waitKey(0)
            cv.destroyWindow('img')
            which_color = 0
            
    cv.destroyAllWindows()
    
