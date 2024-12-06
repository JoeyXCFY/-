import tensorflow as tf
import glob,os
from wnet_qn import xnet
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]='0'

"""
# jpg to png
files = os.listdir('./image')
os.chdir('./image')
for f in files:
    x = os.path.splitext(f)
    if x[1] == ".jpg":  
        name = x[0] + ".png"   
        os.rename(f,name)
os.chdir('../')
"""

parser = argparse.ArgumentParser(description='Remastering')
parser.add_argument('--img_path',   type=str,   default="./image", help='the path of test image list')
parser.add_argument('--result_path',  type=str, default='./bgganResult', help='the path to save results')
args = parser.parse_args()

img_list = sorted(glob.glob(args.img_path+'/*.jpg'))
model_path = './model'
outputdir = args.result_path
factor = 8 # forbid to change

if __name__ == "__main__":
    import os

    count= 0

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    generator = xnet()
    
    checkpoint = tf.train.Checkpoint(
                                     generator=generator,
                                     )   
    checkpoint.restore(tf.train.latest_checkpoint(model_path))  
    generator.trainable = False
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    print('-----bggan-----')

    for n,img in enumerate(img_list):
        # read img
        out_name = os.path.basename(img)
        inputs = tf.io.read_file(img)
        blur_image = tf.io.decode_jpeg(inputs,3)
        h,w,_ = blur_image.shape
        if w%factor!=0:
            blur_image = tf.image.resize(blur_image,[1024,w+factor-int(w)%factor],method=tf.image.ResizeMethod.BICUBIC)
        blur_image = tf.cast(blur_image, tf.float32)
        blur_image = (blur_image / 127.5) - 1.0
        blur_image = tf.expand_dims(blur_image,0)

        # inference
        output =  generator(blur_image)[0]

        # save model
        output = tf.squeeze(output)
        output = (output+1.0)*127.5
        # output = output[0,:h,:w,::-1]
        output = tf.image.resize(output,[h,w],method=tf.image.ResizeMethod.BICUBIC)
        output = tf.cast(output, tf.uint8)
        output = tf.io.encode_jpeg(output)
        tf.io.write_file(os.path.join(outputdir,out_name), output)
        print('processing:',count+1, '/', len(img_list))
        count= count+1

    print('bggan done')