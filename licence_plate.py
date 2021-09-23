import cv2
import os
import math
import glob
import time
import numpy as np
from numpy.linalg import norm

kernel = np.ones((3,3),np.uint8)
kernel_x_1 = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
kernel_x_2 = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
kernel_y_1 = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernel_y_2 = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernel_x = np.ones((3,6),np.uint8)
kernel_y = np.ones((3,3),np.uint8)

SZ = 20  # 训练图片长宽
MAX_WIDTH = 1000  # 原始图片最大宽度
Min_Area = 2000  # 车牌区域允许最大面积
PROVINCE_START = 1000

# 不能保证包括所有省份
provinces = [
	"zh_cuan", "川",
	"zh_e", "鄂",
	"zh_gan", "赣",
	"zh_gan1", "甘",
	"zh_gui", "贵",
	"zh_gui1", "桂",
	"zh_hei", "黑",
	"zh_hu", "沪",
	"zh_ji", "冀",
	"zh_jin", "津",
	"zh_jing", "京",
	"zh_jl", "吉",
	"zh_liao", "辽",
	"zh_lu", "鲁",
	"zh_meng", "蒙",
	"zh_min", "闽",
	"zh_ning", "宁",
	"zh_qing", "靑",
	"zh_qiong", "琼",
	"zh_shan", "陕",
	"zh_su", "苏",
	"zh_sx", "晋",
	"zh_wan", "皖",
	"zh_xiang", "湘",
	"zh_xin", "新",
	"zh_yu", "豫",
	"zh_yu1", "渝",
	"zh_yue", "粤",
	"zh_yun", "云",
	"zh_zang", "藏",
	"zh_zhe", "浙"
]

# 来自opencv的sample，用于svm训练
def preprocess_hog(digits):
	samples = []
	for img in digits:
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		mag, ang = cv2.cartToPolar(gx, gy)
		bin_n = 16
		bin = np.int32(bin_n * ang / (2 * np.pi))
		bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
		mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		hist = np.hstack(hists)

		# transform to Hellinger kernel
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps

		samples.append(hist)
	return np.float32(samples)
	
# 来自opencv的sample，用于svm训练
def deskew(img):
	m = cv2.moments(img)
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	skew = m['mu11'] / m['mu02']
	M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img

	
class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()



def match_color(img):
    y,cr,cb = cv2.split(img);
    
    cb = cv2.dilate(cb,kernel,iterations = 3);
    cb = cv2.erode(cb,kernel,iterations = 3);
    cb = cv2.erode(cb,kernel,iterations = 3);
    cb = cv2.dilate(cb,kernel,iterations = 3);
    
    cr = cv2.dilate(cr,kernel,iterations = 3);
    cr = cv2.erode(cr,kernel,iterations = 3);
    cr = cv2.erode(cr,kernel,iterations = 3);
    cr = cv2.dilate(cr,kernel,iterations = 3);
    
    cb_thres = cv2.inRange(cb,143,190)    #145,190
    cr_thres = cv2.inRange(cr,90,150)    #90,150
    result = cv2.bitwise_and(cb_thres,cr_thres)
    
    return result
    
def match_shape(img):
    org = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contour = cv2.filter2D(org,-1,kernel_x_1) + cv2.filter2D(org,-1,kernel_x_2)
    contour = cv2.medianBlur(contour, 3)
    contour = cv2.inRange(contour,80,255)
    
    contour = cv2.dilate(contour,kernel_x,iterations = 3);
    contour = cv2.erode(contour,kernel_x,iterations = 6);
    contour = cv2.erode(contour,kernel_y,iterations = 1);
    contour = cv2.dilate(contour,kernel_y,iterations = 2);

    return contour;

def boxOverlap(box1, box2):
    if box1[0] >= box2[1]: 
        return 0
    if box1[2] >= box2[3]:  
        return 0 
    if box1[1] <= box2[0]: 
        return 0
    if box1[3] <= box2[2]:
        return 0 
    return 1

def find_licence(org):
    result = org.copy()
    
    color_result = match_color(cv2.cvtColor(org,cv2.COLOR_BGR2YCR_CB))
    shape_result = match_shape(org)
    
    result = cv2.bitwise_and(color_result,shape_result)
    
    cnts,_ = cv2.findContours(color_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    #求框内平均，画方框
    for cnt in cnts:
        rect_tupple = cv2.boundingRect(cnt)
        if rect_tupple[2]>15 and rect_tupple[3]>15:
            if 2 < rect_tupple[2]/rect_tupple[3] < 5:
                x1 = int(rect_tupple[0])
                x2 = int(rect_tupple[0]+rect_tupple[2])
                y1 = int(rect_tupple[1])
                y2 = int(rect_tupple[1]+rect_tupple[3])
                average = np.sum(result[y1:y2,x1:x2])/((x2-x1)*(y2-y1))
                if average>40:
                    boxes.append([x1,x2,y1,y2])
            
    final_boxes = []
    for box1 in boxes:
        for box2 in boxes:
            if boxOverlap(box1,box2)!=0:
                box1[0] = min(box1[0],box2[0])
                box1[1] = max(box1[1],box2[1])
                box1[2] = min(box1[2],box2[2])
                box1[3] = max(box1[3],box2[3])
        final_boxes.append(tuple(box1))
    final_boxes = set(final_boxes)    #去重
    return final_boxes
    
def find_numbers(org):
    result = org.copy()
    cnts,_ = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    #求框内平均，画方框
    for cnt in cnts:
        rect_tupple = cv2.boundingRect(cnt)
        if 170>rect_tupple[3]>100:
            if rect_tupple[2]/rect_tupple[3] < 1:
                x1 = int(rect_tupple[0])
                x2 = int(rect_tupple[0]+rect_tupple[2])
                y1 = int(rect_tupple[1])
                y2 = int(rect_tupple[1]+rect_tupple[3])
                average = np.sum(result[y1:y2,x1:x2])/((x2-x1)*(y2-y1))
                if average>10:
                    boxes.append([x1,x2,y1,y2])
            
    final_boxes = []
    for box1 in boxes:
        for box2 in boxes:
            if boxOverlap(box1,box2)!=0:
                box1[0] = min(box1[0],box2[0])
                box1[1] = max(box1[1],box2[1])
                box1[2] = min(box1[2],box2[2])
                box1[3] = max(box1[3],box2[3])
        final_boxes.append((box1[0],box1[1],box1[2],box1[3]))# x1,x2,y1,y2
    final_boxes = set(final_boxes)    #去重
    return final_boxes

def cal_center(box):
    x = (box[0] + box[1])/2.0
    #y = (box[2] + box[3])/2.0
    y = box[2]
    return (x,y)

model = SVM(C=1, gamma=0.5)            # 字母数字分类器
modelchinese = SVM(C=1, gamma=0.5)    # 中文分类器
if os.path.exists("svm.dat"):
    model.load("svm.dat")
if os.path.exists("svmchinese.dat"):
    modelchinese.load("svmchinese.dat")

#all_files = glob.glob(r'D:\Datasets\CCPD\CCPD_2019_first_part\ccpd_base\*.jpg')
all_files = glob.glob(r'C:\Document\Code\python_licence_plate\*.jpg')

print(all_files)

success = 0
total = 0
for a,file_name in enumerate(all_files):                # for every picture
    t1 = time.time()
    
    #org = cv2.imread(f'{a}.jpg')
    org = cv2.imread(file_name)
    org = cv2.resize(org, (640,math.floor((640/org.shape[1])*org.shape[0])))
    
    licences = find_licence(org)
    total += 1
    if total>=5000:
        break
    for lic in licences:            # for every licence found
        width = lic[1]-lic[0]
        margin = math.floor(width/20)
        # 裁剪出车牌区域
        cropImg = org[lic[2]:lic[3], lic[0]-margin:lic[1]+margin]   # [height,width]
        if cropImg.shape[1]!= 0:
            cropImg = cv2.resize(cropImg, (720,math.floor((720/cropImg.shape[1])*cropImg.shape[0])))
        else:
            continue
        color_binary = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
        contour = cv2.filter2D(color_binary,-1,kernel_x_1) + cv2.filter2D(color_binary,-1,kernel_x_2) + \
                    cv2.filter2D(color_binary,-1,kernel_y_1) + cv2.filter2D(color_binary,-1,kernel_y_2)
        contour = cv2.medianBlur(contour, 3)
        _,contour = cv2.threshold(contour,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # 基于contour
        _,color_binary = cv2.threshold(color_binary,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # adaptive thresholding(fix value don't work)
        result = cv2.bitwise_and(color_binary,contour)                                    # 基于contour + 基于颜色
        result = cv2.dilate(result,kernel,iterations = 1);                                # 加粗白色边缘                
        
        
        #cv2.imshow(f'{a}',cropImg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        
        # 找出数字，计算旋转角度
        number_boxes = find_numbers(result) 
        centers = []
        for box in number_boxes: 
            centers.append(cal_center(box))
            cv2.rectangle(result, (box[0], box[2]), (box[1], box[3]), 150, thickness = 2) #字符区域画框
        centers.sort(key = lambda x:x[1])
        
        if len(centers)>=3:        # can recognize 3 characters
            middle = math.floor(len(centers)/2)
            valid_centers = centers[middle-1:middle+2]
            degree = (math.atan((valid_centers[0][1]-valid_centers[1][1])/(valid_centers[0][0]-valid_centers[1][0]))*(180/math.pi)+\
                math.atan((valid_centers[1][1]-valid_centers[2][1])/(valid_centers[1][0]-valid_centers[2][0]))*(180/math.pi))/2
            
            (h,w) = cropImg.shape[:2]
            center = (w / 2,h / 2)
            M = cv2.getRotationMatrix2D(center,degree,1)# 旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
            cropImg = cv2.warpAffine(cropImg,M,(w,h))
            
            # 然后从头计算一遍，抓出所有数字
            color_binary = cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY)
            contour = cv2.filter2D(color_binary,-1,kernel_x_1) + cv2.filter2D(color_binary,-1,kernel_x_2) + \
                        cv2.filter2D(color_binary,-1,kernel_y_1) + cv2.filter2D(color_binary,-1,kernel_y_2)
            contour = cv2.medianBlur(contour, 3)
            _,contour = cv2.threshold(contour,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # 基于contour
            _,color_binary = cv2.threshold(color_binary,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # adaptive thresholding(fix value don't work)
            result = cv2.bitwise_and(color_binary,contour)                                    # 基于contour + 基于颜色
            result = cv2.dilate(result,kernel,iterations = 1);                                # 加粗白色边缘                
            
            
            # 找出数字，计算上下边界
            number_boxes = find_numbers(result) 
            upper = []
            lower = []
            for box in number_boxes: 
                cv2.rectangle(result, (box[0], box[2]), (box[1], box[3]), 150, thickness = 2) #字符区域画框
                upper.append(box[2])
                lower.append(box[3])
            upper.sort()
            lower.sort()
            middle = math.floor(len(upper)/2)
            up_bond = upper[middle]
            low_bond = lower[middle]
            margin = math.floor((low_bond-up_bond)/20)
            cropped = cropImg[up_bond-margin:low_bond+margin, :]    # y , x
            
            cropped = cv2.resize(cropped, (math.floor((160/cropped.shape[0])*cropped.shape[1]),160))
            
            # 然后从头计算一遍，抓出所有数字
            color_binary = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            contour = cv2.filter2D(color_binary,-1,kernel_x_1) + cv2.filter2D(color_binary,-1,kernel_x_2) + \
                        cv2.filter2D(color_binary,-1,kernel_y_1) + cv2.filter2D(color_binary,-1,kernel_y_2)
            contour = cv2.medianBlur(contour, 3)
            _,contour = cv2.threshold(contour,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # 基于contour
            _,color_binary = cv2.threshold(color_binary,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)    # adaptive thresholding(fix value don't work)
            result = cv2.bitwise_and(color_binary,contour)                                    # 基于contour + 基于颜色
            result = cv2.dilate(result,kernel,iterations = 1);                                # 加粗白色边缘                
            
            # 找出数字
            number_boxes = find_numbers(result)
            number_boxes = sorted(number_boxes,key = lambda x: x[0])
            
            x_value = []
            x_width = []
            for box in number_boxes:         
                x_value.append(cal_center(box)[0])
                x_width.append(box[1]-box[0])
            
            try:
                gap = [(x_value[i+1]-x_value[i]) for i in range(len(x_value)-1)] # 找到中心点间隔最大位置（2-5分界点）
                gap_index = gap.index(max(gap))
                interval = x_value[gap_index+2]-x_value[gap_index+1]
                x_width.sort()
                width = x_width[math.floor(len(x_width)/2)]
                if width<50:
                    width = 74
                #print(f'interval: {interval}, width: {width}')
            
            
                cuts = []
                cuts.append((math.floor(x_value[gap_index]-interval-width/2),math.floor(x_value[gap_index]-interval+width/2)))#第1位
                cuts.append((math.floor(x_value[gap_index]-width/2),math.floor(x_value[gap_index]+width/2)))#第2位
                cuts.append((math.floor(x_value[gap_index+1]-width/2),math.floor(x_value[gap_index+1]+width/2)))#第3位
                cuts.append((math.floor(x_value[gap_index+2]-width/2),math.floor(x_value[gap_index+2]+width/2)))#第4位
                cuts.append((math.floor(x_value[gap_index+3]-width/2),math.floor(x_value[gap_index+3]+width/2)))#第5位
                cuts.append((math.floor(x_value[gap_index+4]-width/2),math.floor(x_value[gap_index+4]+width/2)))#第6位
                cuts.append((math.floor(x_value[gap_index+5]-width/2),math.floor(x_value[gap_index+5]+width/2)))#第7位
                
                thres = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                _,thres = cv2.threshold(thres,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                plate = ''
                for i, cut in enumerate(cuts):
                    cv2.rectangle(cropped, (cut[0], 0), (cut[1], 159), (0,255,0), thickness = 2) #字符区域画框
                    
                    part_card = thres[0:159, cut[0]:cut[1]]
                    w = abs(part_card.shape[1] - SZ) // 2
                    part_card = cv2.copyMakeBorder(part_card , 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
                    part_card = preprocess_hog([part_card])
                    if i == 0:
                        resp = modelchinese.predict(part_card)
                        charactor = provinces[int(resp[0]) - PROVINCE_START]
                    else:
                        resp = model.predict(part_card)
                        charactor = chr(resp[0])
                    plate = plate + charactor
                
                success += 1
                cv2.imshow(f'{a}',cropped)
                print(plate)
                print('org', org.shape)
                print('cropped', cropped.shape)
                cv2.waitKey(0);
                cv2.destroyAllWindows()
                
            except Exception as e:
                print(f'{a} recognize error')
        else:                    
            print(f'{a} invalid plate')
        
    t2 = time.time()
    print(f'{a} time taken:{t2-t1}')
print(success,' ',total)
'''    
#contour = cv2.equalizeHist(contour)  
B,G,R = cv2.split(cropImg)
        
#B = cv2.erode(B,kernel,iterations = 3);
#G = cv2.erode(G,kernel,iterations = 3);
#R = cv2.erode(R,kernel,iterations = 3);

cropImg = cv2.merge([B,G,R])
#cv2.rectangle(org, (box[0], box[2]), (box[1], box[3]), (0,255,0), thickness = 2) #车牌区域画框

contour = cv2.dilate(contour,kernel_x,iterations = 3);
contour = cv2.erode(contour,kernel_x,iterations = 6);
contour = cv2.dilate(contour,kernel_x,iterations = 3);

contour = cv2.erode(contour,kernel_y,iterations = 1);
contour = cv2.dilate(contour,kernel_y,iterations = 2);
'''
'''
if rect_tupple[2]>=-10 or rect_tupple[2]<=-80:
            if rect_tupple[1][1]!=0 and rect_tupple[1][0]!=0:
                if rect_tupple[1][0] >= rect_tupple[1][1]:
                    w = rect_tupple[1][0] * math.cos(rect_tupple[2])
                    h = rect_tupple[1][1] * math.cos(rect_tupple[2])
                    print('1', rect_tupple[1],' ',rect_tupple[2], w,'  ',h)
                else:
                    w = rect_tupple[1][1] * math.sin(abs(rect_tupple[2]))
                    h = rect_tupple[1][0] * math.sin(abs(rect_tupple[2]))
                    print('2', rect_tupple[1],' ',rect_tupple[2], w,'  ',h)
                
                if w/h > 2.5:
                    rect_vertices = cv2.boxPoints( rect_tupple )
                    rect_vertices = np.int0(rect_vertices)
                    cv2.drawContours(org,[rect_vertices],-1,(0,255,0),thickness = 2)
                
                    
'''

'''
for cnt in cnts:
    rect_tupple = cv2.minAreaRect( cnt )
    if rect_tupple[1][1]>10 and rect_tupple[1][0]>10:
        if 5 > rect_tupple[1][0]/rect_tupple[1][1] > 2 or 5 > rect_tupple[1][1]/rect_tupple[1][0] > 2:
            rect_vertices = cv2.boxPoints( rect_tupple )
            rect_vertices = np.int0(rect_vertices)
            cv2.drawContours(canvas,[rect_vertices],-1,(0,255,0),thickness = 2)



for cnt in cnts:
    rect_tupple = cv2.minAreaRect( cnt )
    rect_vertices = cv2.boxPoints( rect_tupple )
    rect_vertices = np.int0(rect_vertices)
    cv2.drawContours(canvas,[rect_vertices],-1,(0,255,0),thickness = 2)
'''



#org = cv2.GaussianBlur(org,(5,5),cv2.BORDER_DEFAULT)
#contour = cv2.Canny(org, 100, 200)

#print(org.shape)
#统一图片尺寸1920*xxxx
#org = cv2.erode(org,kernel,iterations = 2);
#org = cv2.dilate(org,kernel,iterations = 2);
#
#
#
