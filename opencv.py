import cv2, sys
import glob

path = "image"

def img_resize(image):
    height, width = image.shape[0], image.shape[1]

    width_new = 640
    height_new = 480

    if width / height >= width_new /height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new /width)))
    else:
        img_new =cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new


def openCVbokeh(img):
    global ap

    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors= 5)

    print("Detected Faces: ", faces)

    ksize =int(img.shape[0]/ap) 

    if (ksize%2) == 0 :
        ksize = ksize + 1

    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)

    for face in faces:
        face_img = img[face[1]:face[1] + face[3], face[0]:face[0]+ face[2]]

        blur[face[1]:face[1]+ face[3], face[0]:face[0]+ face[2]] = face_img
        #blur_new = img_resize(blur)
        blur_new = blur
    
    return blur_new


if __name__ == "__main__":
    
    pa = glob.glob(f"{path}/*.jpg")
    path_dir="./opencvResult/" 
    count=0

    print('-----openCV-----')

    for img_name in pa:
        color_image = cv2.imread(img_name)
        for i in range(4):
            global ap
            api=[20,40,80,200]
            ap= api[i]
            image = openCVbokeh(color_image)
            cv2.imwrite(path_dir+'%d.jpg'%(count), image)
            print('processing:',count+1, '/', len(pa)*len(range(4)))
            count=count+1

    
    print('openCV done')
    