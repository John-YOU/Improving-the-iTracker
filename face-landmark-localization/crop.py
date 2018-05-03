#usage :python landmarkPredict.py predictImage  testList.txt
# the point: 
#0-15, left face edge (top to bottom)
#16-17, the bottom point
#18-33, right face edge (bottom to top)
#34-43, left eye brow (left to right)
#44-53, right eye brow (left to right)
#54-72,nose
#73-96, two eyes;
#97-136, the mouth 
#to get the most left point: select from 0-13
#to get the most right point: select from 20-33
#to get the bottom point: select from 8-25
#to get the top point: select from 34-53     
import os
import sys
import numpy as np
import cv2
import caffe
import dlib
import matplotlib.pyplot as plt
from math import sqrt
import glob

system_height = 3000
system_width = 5000
channels = 1
test_num = 1
pointNum = 68

S0_width = 60
S0_height = 60
vgg_height = 224
vgg_width = 224
M_left = -0.15
M_right = +1.15
M_top = -0.10
M_bottom = +1.25
pose_name = ['Pitch', 'Yaw', 'Roll']     # respect to  ['head down','out of plane left','in plane right']
 #record the current face number
currentImage=""

def recover_coordinate(largetBBox, facepoint, width, height):
    point = np.zeros(np.shape(facepoint))
    cut_width = largetBBox[1] - largetBBox[0]
    cut_height = largetBBox[3] - largetBBox[2]
    scale_x = cut_width*1.0/width;
    scale_y = cut_height*1.0/height;
    point[0::2]=[float(j * scale_x + largetBBox[0]) for j in facepoint[0::2]]
    point[1::2]=[float(j * scale_y + largetBBox[2]) for j in facepoint[1::2]]
    return point

def show_image(img, facepoint, bboxs, headpose, file_index):
    f_num = 0
    plt.figure(figsize=(20,10))
    for faceNum in range(0,facepoint.shape[0]):
        #store the eye in the left side of image
        (x1,y1,x2,y2)=[facepoint[faceNum, 72],facepoint[faceNum, 73],facepoint[faceNum, 78],facepoint[faceNum, 79]]
        dist = sqrt( (x1 - x2)**2 + (y1 - y2)**2 ) #length of eye
        xdist=(1.6*dist-(x2-x1))/2 #the  distance that should expand
        top_y=min([y1,y2,facepoint[faceNum, 75],facepoint[faceNum, 77]])
        bottom_y=max([y1,y2,facepoint[faceNum, 81],facepoint[faceNum, 83]])
        ydist=(1.6*dist-(bottom_y-top_y))/2
        (e1Left, e1Right, e1Top, e1Bottom)=[ max([int(round(x1-xdist)), 0]), min([int(round(x2+xdist)),img.shape[1]]), max([int(round(top_y-ydist)), 0]), min([int(round(bottom_y+ydist)), img.shape[0]]) ]
        e1Right=e1Left+(e1Bottom-e1Top)
        eye1=img[e1Top:e1Bottom,e1Left:e1Right]
        eye1=cv2.resize(eye1,(224,224))
        cv2.imwrite(e1Path+currentImage[:-4]+"f"+str(max(faceNum,file_index))+"eye1.jpg",eye1)

        #store the eye in the right side of image
        (x1,y1,x2,y2)=[facepoint[faceNum, 84],facepoint[faceNum, 85],facepoint[faceNum, 90],facepoint[faceNum, 91]]
        dist = sqrt( (x1 - x2)**2 + (y1 - y2)**2 ) #length of eye
        xdist=(1.6*dist-(x2-x1))/2 #the  distance that should expand
        top_y=min([y1,y2,facepoint[faceNum, 87],facepoint[faceNum, 89]])
        bottom_y=max([y1,y2,facepoint[faceNum, 93],facepoint[faceNum, 95]])
        ydist=(1.6*dist-(bottom_y-top_y))/2
        (e2Left, e2Right, e2Top, e2Bottom)=[ max([int(round(x1-xdist)), 0]), min([int(round(x2+xdist)),img.shape[1]]), max([int(round(top_y-ydist)), 0]), min([int(round(bottom_y+ydist)), img.shape[0]]) ]
        e1Left=e1Right-(e1Bottom-e1Top)
        eye2=img[e2Top:e2Bottom,e2Left:e2Right]
        eye2=cv2.resize(eye2,(224,224))
        cv2.imwrite(e2Path+currentImage[:-4]+"f"+str(max(faceNum,file_index))+"eye2.jpg",eye2)

        #rescale and store the face if the after scale the new box still won't beyond the border, else just use the original
        fTop=int(round(min([facepoint[faceNum, i] for i in range(35,54,2)])-dist*1.4)) # to include forehead
        fBottom=int(round(max([facepoint[faceNum, i] for i in range(9,26,2)])))
        leng=fBottom-fTop
        fLeft=min([facepoint[faceNum, i] for i in range(0,13,2)])
        fRight=max([facepoint[faceNum, i] for i in range(20,33,2)])
        hdist=(leng-(fRight-fLeft))/2
        fLeft=int(round(fLeft-hdist))
        fRight=fLeft+leng # to mkae sure get a square
        #f_num=f_num+1

        if fLeft>=0 and fRight<=img.shape[1] and fTop>=0 and fBottom<=img.shape[0]:#the new box is valid:
            #cv2.rectangle(img, (fLeft, fTop), (fRight,fBottom), (0,0,255), 2)
            cface=img[fTop:fBottom,fLeft:fRight]
            cv2.imwrite(fPath+"face"+str(max(faceNum,file_index))+'.jpg',cface)
        else:
            #cv2.rectangle(img, (int(bboxs[faceNum,0]), int(bboxs[faceNum,2])), (int(bboxs[faceNum,1]), int(bboxs[faceNum,3])), (0,0,255), 2)
            cface=img[int(bboxs[faceNum,2]):int(bboxs[faceNum,3]),int(bboxs[faceNum,0]):int(bboxs[faceNum,1])]
            cv2.imwrite(fPath+"face"+str(max(faceNum,file_index))+'.jpg',cface)
        '''for p in range(0,3):
            plt.text(int(bboxs[faceNum,0]), int(bboxs[faceNum,2])-p*30,
                '{:s} {:.2f}'.format(pose_name[p], headpose[faceNum,p]),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=12, color='white')'''
        #for i in range(0,facepoint.shape[1]/2):
        #    cv2.circle(img,(int(round(facepoint[faceNum,i*2])),int(round(facepoint[faceNum,i*2+1]))),1,(0,255,0),2)
    	##get the face grid of the face
    	gridLX=int(fLeft*25.0/img.shape[1])
    	gridLY=int(fTop*25.0/img.shape[0])
    	gridHX=int(round(fRight*25.0/img.shape[1]+0.5))
    	gridHY=int(round(fBottom*25.0/img.shape[0]+0.5))
    	print (gridLX,gridLY,gridHX,gridHY)
    	grid=""
    	for i in range(0,25):
    		for j in range(0,25):
    			if i>gridLY and i<gridHY:
    				if j>gridLX and j<gridHX:
    					grid=grid+"1 "
    					continue
    			grid=grid+"0 "
    	g=open(gPath+"face_grid"+str(max(faceNum,file_index)),"w")
    	g.write(grid)
    	g.close()


    height = img.shape[0]
    width = img.shape[1]
    if height > system_height or width > system_width:
        height_radius = system_height*1.0/height
        width_radius = system_width*1.0/width
        radius = min(height_radius,width_radius)
        img = cv2.resize(img, (0,0), fx=radius, fy=radius)
    #img = img[:,:,[2,1,0]]
    #cv2.imwrite('/Users/youhan/Documents/test/testEye/whole'+str(max(faceNum,file_index))+'.jpg',img[:,:,::-1])
    #plt.imshow(img)
    #plt.show()


def recoverPart(point,bbox,left,right,top,bottom,img_height,img_width,height,width):
    #(136 points, facebox_lrtb, M_left, M_right, M_top, M_bottom, imgH, imgW, faceH, faceW)
    largeBBox = getCutSize(bbox,left,right,top,bottom)
    retiBBox = retifyBBoxSize(img_height,img_width,largeBBox)
    #retiBBox=bbox
    recover = recover_coordinate(retiBBox,point,height,width)
    recover=recover.astype('float32')
    return recover


def getRGBTestPart(bbox,left,right,top,bottom,img,height,width):
    #print bbox
    largeBBox = getCutSize(bbox,left,right,top,bottom)
    retiBBox = retifyBBox(img,largeBBox)
    #retiBBox=bbox
    #print retiBBox
    # cv2.rectangle(img, (int(retiBBox[0]), int(retiBBox[2])), (int(retiBBox[1]), int(retiBBox[3])), (0,0,255), 2)
    # cv2.imshow('f',img)
    # cv2.waitKey(0)
    face = img[int(retiBBox[2]):int(retiBBox[3]), int(retiBBox[0]):int(retiBBox[1]), :]
    face = cv2.resize(face,(height,width),interpolation = cv2.INTER_AREA)
    face=face.astype('float32')
    #cv2.imwrite("/Users/youhan/Documents/face-landmark-localization-master/face.jpg",face)
    return face

def batchRecoverPart(predictPoint,totalBBox,totalSize,left,right,top,bottom,height,width):
    #(136 points, facebox_lrtb, imgSIZE, M_left, M_right, M_top, M_bottom,faceH, faceW)
    recoverPoint = np.zeros(predictPoint.shape)
    for i in range(0,predictPoint.shape[0]):
        recoverPoint[i] = recoverPart(predictPoint[i],totalBBox[i],left,right,top,bottom,totalSize[i,0],totalSize[i,1],height,width)
    return recoverPoint



def retifyBBox(img,bbox):
    img_height = np.shape(img)[0] - 1
    img_width = np.shape(img)[1] - 1
    if bbox[0] <0:
        bbox[0] = 0
    if bbox[1] <0:
        bbox[1] = 0
    if bbox[2] <0:
        bbox[2] = 0
    if bbox[3] <0:
        bbox[3] = 0
    if bbox[0] > img_width:
        bbox[0] = img_width
    if bbox[1] > img_width:
        bbox[1] = img_width
    if bbox[2]  > img_height:
        bbox[2] = img_height
    if bbox[3]  > img_height:
        bbox[3] = img_height
    return bbox

def retifyBBoxSize(img_height,img_width,bbox):
    if bbox[0] <0:
        bbox[0] = 0
    if bbox[1] <0:
        bbox[1] = 0
    if bbox[2] <0:
        bbox[2] = 0
    if bbox[3] <0:
        bbox[3] = 0
    if bbox[0] > img_width:
        bbox[0] = img_width
    if bbox[1] > img_width:
        bbox[1] = img_width
    if bbox[2]  > img_height:
        bbox[2] = img_height
    if bbox[3]  > img_height:
        bbox[3] = img_height
    return bbox

def getCutSize(bbox,left,right,top,bottom):   #left, right, top, and bottom
    box_width = bbox[1] - bbox[0]
    box_height = bbox[3] - bbox[2]
    cut_size=np.zeros((4))
    cut_size[0] = bbox[0] + left * box_width
    cut_size[1] = bbox[1] + (right - 1) * box_width
    cut_size[2] = bbox[2] + top * box_height
    cut_size[3] = bbox[3] + (bottom-1) * box_height
    return cut_size


def detectFace(img):
    detector = dlib.get_frontal_face_detector()
    dets = detector(img,1)
    bboxs = np.zeros((len(dets),4))
    for i, d in enumerate(dets):
        bboxs[i,0] = d.left();
        bboxs[i,1] = d.right();
        bboxs[i,2] = d.top();
        bboxs[i,3] = d.bottom();
    return bboxs;


def predictImage(filelist):
    vgg_point_MODEL_FILE = 'model/deploy.prototxt'
    vgg_point_PRETRAINED = 'model/68point_dlib_with_pose.caffemodel'
    mean_filename='model/VGG_mean.binaryproto'
    vgg_point_net=caffe.Net(vgg_point_MODEL_FILE,vgg_point_PRETRAINED,caffe.TEST)
    caffe.set_mode_cpu()
    #caffe.set_mode_gpu()
    #caffe.set_device(0)

    proto_data = open(mean_filename, "rb").read()
    a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
    mean = caffe.io.blobproto_to_array(a)[0]
    file_index=0

    for info in filelist:
        print file_index
        imgPath = info
        currentImage=imgPath # for the naming of eyes
        num = 1
        colorImage = cv2.imread(imgPath)
        bboxs = detectFace(colorImage) # (10, 4) means 10 faces
        faceNum = bboxs.shape[0] # num of faces
        faces = np.zeros((1,3,vgg_height,vgg_width)) #dimension 1 3 224 224, the first refers to different faces....
        predictpoints = np.zeros((faceNum,pointNum*2)) #dimension (num of face, points * 2)
        predictpose = np.zeros((faceNum,3)) # three kinds of pose
        imgsize = np.zeros((2)) # (2,)
        imgsize[0] = colorImage.shape[0]-1  #height-1
        imgsize[1] = colorImage.shape[1]-1 #width-1 
        TotalSize = np.zeros((faceNum,2)) #(num of face, 2)
        for i in range(0,faceNum): 
            TotalSize[i] = imgsize # num of face * (h, w)
        for i in range(0,faceNum):
            bbox = bboxs[i] # (left, right, top, bottom)
            colorface = getRGBTestPart(bbox,M_left,M_right,M_top,M_bottom,colorImage,vgg_height,vgg_width) # 3 224 224
            normalface = np.zeros(mean.shape) # 3 224 224
            normalface[0] = colorface[:,:,0]
            normalface[1] = colorface[:,:,1]
            normalface[2] = colorface[:,:,2]
            normalface = normalface - mean
            faces[0] = normalface

            blobName = '68point'
            data4DL = np.zeros([faces.shape[0],1,1,1])
            vgg_point_net.set_input_arrays(faces.astype(np.float32),data4DL.astype(np.float32))
            vgg_point_net.forward()
            predictpoints[i] = vgg_point_net.blobs[blobName].data[0] # (136,)

            blobName = 'poselayer'
            pose_prediction = vgg_point_net.blobs[blobName].data #(1,3)
            predictpose[i] = pose_prediction * 50 # (3, )

        predictpoints = predictpoints * vgg_height/2 + vgg_width/2
        level1Point = batchRecoverPart(predictpoints,bboxs,TotalSize,M_left,M_right,M_top,M_bottom,vgg_height,vgg_width)
        show_image(colorImage, level1Point, bboxs, predictpose,file_index)        
        file_index = file_index + 1

path='/Users/youhan/Documents/FYP/mapping_data/you-11-28/test2/'
flist=glob.glob(path+"/raw/*")
flist=sorted(flist)
e1Path=path+"/eyeRight/"
e2Path=path+"/eyeLeft/"
fPath=path+"/face/"
gPath=path+"/grid/"
if not os.path.exists(e1Path):
    os.makedirs(e1Path)
if not os.path.exists(e2Path):
    os.makedirs(e2Path)
if not os.path.exists(fPath):
    os.makedirs(fPath)
if not os.path.exists(gPath):
    os.makedirs(gPath)

predictImage(flist)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
    else:
        func = globals()[sys.argv[1]]
        func(*sys.argv[2:])