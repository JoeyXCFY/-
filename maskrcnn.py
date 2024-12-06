import numpy as np
import cv2
import torch
import os
import glob
import torch
from train import get_model_instance_segmentation, get_transform
import transforms as T
import torchvision
import math


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = "image"

def blurbox(model, img):
    # 輸入的img是0-1範圍的tensor
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    with torch.no_grad():
        '''
        prediction形如：
        [{'boxes': tensor([[1492.6672,  238.4670, 1765.5385,  315.0320],
        [ 887.1390,  256.8106, 1154.6687,  330.2953]], device='cuda:0'), 
        'labels': tensor([1, 1], device='cuda:0'), 
        'scores': tensor([1.0000, 1.0000], device='cuda:0')}]
        '''
        prediction = model([img.to(device)])

    # print(prediction)

    img = img.permute(1, 2, 0)  # C,H,W → H,W,C，用來畫圖
    img = (img * 255).byte().data.cpu()  # * 255，float轉0-255
    img = np.array(img)  # tensor → ndarray
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.array(prediction[0]['masks'].detach().cpu() * 255)
    mask = mask.astype("uint8")
    mb = np.zeros(img.shape, dtype=img.dtype) 
    mm = mask[0][0]
    temp = np.zeros(mm.shape, dtype=mm.dtype)

    for i in range(prediction[0]['boxes'].cpu().shape[0]):
        xmin = round(prediction[0]['boxes'][i][0].item())
        ymin = round(prediction[0]['boxes'][i][1].item())
        xmax = round(prediction[0]['boxes'][i][2].item())
        ymax = round(prediction[0]['boxes'][i][3].item())
        label = prediction[0]['labels'][i].item()
        mm = mask[i][0]
        mr = cv2.bitwise_or(mm,temp)
        temp = mm.copy()
        
    if label == 1:
        contours, hierarchy = cv2.findContours(mr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.fillPoly(mb, [contours][0], (255, 255, 255))
        
        if img.shape[0] >= img.shape[1] :
            sp = img.shape[0]
        else :
            sp = img.shape[1]
        
        ksize = [int(sp/ap), int(sp/(ap*1.5)), int(sp/(ap*5)), int(sp/(ap*12.5))]
        for i in range(len(ksize)) :
            if ksize[i]%2 == 0 :
                ksize[i] = ksize[i]+ 1

        kernel = np.ones((ksize[1],ksize[1]), np.uint8)
        kernel2 = np.ones((ksize[1],ksize[1]), np.uint8)
        erosion2 = cv2.erode(mb, kernel2, iterations = 1)
        erosion = cv2.erode(mb, kernel, iterations = 1)
        dilation = cv2.dilate(mb, kernel, iterations = 1)
            
        """
        blurbox img:
        body_img = img[ymin: ymax, xmin: xmax]

        blur[ymin: ymax, xmin: xmax] = body_img
        """
        #blurmask

        #dilation
        tempImg_d = img.copy()
        tempImg_d = cv2.GaussianBlur(tempImg_d,(ksize[0],ksize[0]), 0)
        mask_inv_d = cv2.bitwise_not(dilation)
        img1_bg_d = cv2.bitwise_and(img,dilation)
        img2_fg_d = cv2.bitwise_and(tempImg_d,mask_inv_d)

        #org
        tempImg = img.copy()
        tempImg = cv2.GaussianBlur(tempImg, (ksize[1],ksize[1]), 0)
        mask_inv = cv2.bitwise_not(mb)
        img1_bg = cv2.bitwise_and(img,mb)
        img2_fg = cv2.bitwise_and(tempImg,mask_inv)
        img2_fgr = cv2.bitwise_and(img2_fg,dilation)
        blur0 = cv2.add(img2_fgr,img2_fg_d)

        #erosion
        tempImg_e = img.copy()
        tempImg_e = cv2.GaussianBlur(tempImg_e, (ksize[2],ksize[2]), 0)
        mask_inv_e = cv2.bitwise_not(erosion)
        img1_bg_e = cv2.bitwise_and(img,erosion)
        img2_fg_e = cv2.bitwise_and(tempImg_e,mask_inv_e)
        img2_fg_er = cv2.bitwise_and(img2_fg_e,mb)
        blur1 = cv2.add(img2_fg_er,blur0)


        #erosion2
        tempImg_e2 = img.copy()
        tempImg_e2 = cv2.GaussianBlur(tempImg_e2, (ksize[3],ksize[3]), 0)
        mask_inv_e2 = cv2.bitwise_not(erosion2)
        img1_bg_e2 = cv2.bitwise_and(img,erosion2)
        img2_fg_e2 = cv2.bitwise_and(tempImg_e2,mask_inv_e2)
        img2_fg_er2 = cv2.bitwise_and(img2_fg_e2,erosion)
        blur2 = cv2.add(img2_fg_er2,blur1)

        blur = cv2.add(img1_bg_e2, blur2)
        
        #final process
        tempImg_f = blur.copy()
        tempImg_f = cv2.GaussianBlur(tempImg_f, (ksize[2], ksize[2]), 0)
        mask_inv_f = cv2.bitwise_not(erosion2)
        img1_bg_f = cv2.bitwise_and(blur,erosion2)
        img2_fg_f = cv2.bitwise_and(tempImg_f,mask_inv_f)
        blur_f = cv2.add(img1_bg_f,img2_fg_f)
        """
        tempImg = img.copy()
        tempImg = cv2.GaussianBlur(tempImg, (ksize[0], ksize[0]), 0)
        mask_inv = cv2.bitwise_not(mb)
        img1_bg = cv2.bitwise_and(img,mb)
        img2_fg = cv2.bitwise_and(tempImg,mask_inv)
        blur = cv2.add(img1_bg,img2_fg)


        m = cv2.moments(mb[:,:,1])
        # a random point chosen
        # to calculate radius
        k=mm[0]
        # Add your Focus cordinates here
        # m10/m00 and m01/m00 are the center of the mask 
        fx,fy = int(m['m10']/m['m00']), int(m['m01']/m['m00']) 
        dist = math. sqrt((k[0]-fx)**2 + (k[1]-fy)**2)
        # Standard Deviation of the Gaussian
        sigma = dist 
        rows,cols = img.shape[:2]
        potrait = np.copy(blur)
        potrait[:,:,:] = 0
        a = cv2.getGaussianKernel(2*cols ,sigma)[cols-fx:2*cols-fx]
        b = cv2.getGaussianKernel(2*rows ,sigma)[rows-fy:2*rows-fy]
        c = b*a.T
        d = c/c.max()
        potrait[:,:,0] = blur[:,:,0]*d
        potrait[:,:,1] = blur[:,:,1]*d
        potrait[:,:,2] = blur[:,:,2]*d
        """

    # plt.figure(figsize=(20, 15))
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    img = cv2.cvtColor(blur_f, cv2.COLOR_RGB2BGR)
    #print("contours ", contours)
    return img

def blur_show(img):
    transform1 = T.Compose([
        T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    num_class = 2
    model = get_model_instance_segmentation(num_class)
    #model.load_state_dict(torch.load("test.pth"))
    model.load_state_dict(torch.load("test.pth", map_location='cpu'))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    xx, _ = transform1(img, 0)
    xx = blurbox(model, xx)
    return xx


if __name__ == '__main__':

    pa = glob.glob(f"{path}/*.jpg")
    path_dir="./maskrcnnResult/" 
    count=0

    print('-----maskRCNN-----')

    for img_name in pa:
        color_image = cv2.imread(img_name)
        for i in range(4):
            global ap
            api=[20,40,80,200]
            ap= api[i]
            image = blur_show(color_image)
            cv2.imwrite(path_dir+'%d.jpg'%(count), image)
            print('processing:',count+1, '/', len(pa)*len(range(4)))
            count=count+1

        
        
    
    print('maskRCNN done')
        #cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        #cv2.imshow('RealSense', image)
        #cv2.waitKey(0)

