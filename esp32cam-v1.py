import cv2 as cv
import numpy as np


#----------------------------------------------STAGE-1------------------------------------------------------------------------------------------------
# Image input
cap = cv.VideoCapture('https://ce22-103-98-38-17.ngrok-free.app/video')
#-----------------------------------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------STAGE-3(I)---------------------------------------------------------------------------------------------
whT = 320               #stand for "width", "height", "Target".We use 320 becuase the size of config adnd weight fiel is 320

confThreshold = 0.3
nmsThreshold = 0.3         # lower the number lesser the number of bounding boxes for an object


#---------------------------------------------STAGE-2-------------------------------------------------------------------------------------------------
# Initialize COCO dataset 
classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
              "fire hydrant","stop sign","parking m","bird","cat","dog","horse","sheep","cow","elephant",
              "bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
              "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
              "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
              "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
              "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",
              "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",
              "teddy bear","hair drier","toothbrush"]

# Initialize the Weight and Configuration file
modelConfiguration = 'YOLOv3\yolov3.cfg'
modelWeights = 'YOLOv3\yolov3.weights'

# Creat the neural network of YOLO
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

# Have to declare that we use OpenCV and CPU as backend for processing
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
#-----------------------------------------------------------------------------------------------------------------------------------------------------


#------------------------------------------STAGE-6-----------------------------------------------------------------------------------------------------
# Function for getting the detections from the output layer fo the network to the image
def findObjects(outputs, frame):
    height, width, chnl = frame.shape
    print("dimensions taken for w, h:" + str(height)+ ", " + str(height))
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            # print(det[0:5])
            # first5 = det.shape
            # print(first5)
            scores = det[5:]        #In Python, when you slice a list or a string, you can specify the range of indices you want to extract using square brackets []. The colon : is used to indicate a range of indices. In this case, detection[5:] means that you want to extract elements from index 5 to the end of the detection list or string.
            classId = np.argmax(scores)
            confidence = scores[classId] 
            if confidence > confThreshold:
                w, h = int(det[2]*width), int(det[3]*height)
                # print("dimensions of detection: "+ str(w) + "," + str(h))
                # print(w, h)
                x,y = int((det[0]*width)-w/2), int((det[1]*height)-h/2)
                # x,y = int((det[0]*width)-(width/2)), int((det[1]*height/100)-(height/2))
                # print("detection coordinates:" + str(x) + ", " + str(y))
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    print(len(bbox))

# Now we have to eleminate the multiple detection for an object by using a pre-built function known as 
# 'maximum supression" which will get the bounding box of maximum confidence number.

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    print(indices)
    for i in indices:
        # i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0, 255),1)
        cv.putText(frame, f'{classNames[classIds[i]]}{int(confs[i]*100)}%', (max(0,x),max(30,y)), cv.FONT_HERSHEY_SIMPLEX, .65,(255,0,255),1)
    
    # return frame
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Processing
while True:
    ignore, frame = cap.read()
    img_size = frame.shape[:2]
    print(img_size)

    # ignore, ori = cap.read()
    # img_size = ori.shape[:2]
    # frame = cv.resize(ori,(int(img_size[1]/2), int(img_size[0]/2)) , interpolation=cv.INTER_AREA)

    # print("original imge size" + str(img_size))
    # f_height = img_size[0]
    # f_width = img_size[1]
  
    # frame = cv.resize(ori,(int(f_width/2), int(f_height/2)) , interpolation=cv.INTER_AREA)
    print("resized frame:" + str(frame.shape))


#------------------------------------STAGE-3(II)-------------------------------------------------------------------------------------------------------------------------------

# Customize the image as Blob, so that in can be use in the neural network. DNN accept particular type of format known as "Blob"
#a blob is just a (potentially collection) of image(s) with the same spatial dimensions (i.e., width and height), same depth 
# (number of channels), that have all be preprocessed in the same manner.
    blob = cv.dnn.blobFromImage(frame, 1/255,(int(img_size[1]/2), int(img_size[1]/2)), [0,0,0],1, crop=False)

# Set the image in network
    net.setInput(blob)

# In neural network there are 3 different output layers. 1st we have to know the names of these layers
    layerNames = net.getLayerNames()  #get the names of all the layer
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------STAGE-4------------------------------------------------------------------------------------------------------------------------------------
    #print the names of all the layers
    # print(layerNames) 
    
    #print the id of the three output layers
    # print(net.getUnconnectedOutLayers())
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------STAGE-5------------------------------------------------------------------------------------------------------------------------------------
   # Now we should refer to the layer name from the output layer id. Also layer count no. is always 1 greater than the original output no.
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    print(outputNames)          #print the names of the 3 output layers


    # Now we can send the input image as forward pass to the network and find the results of these 3 layers in the image
    #"The forward pass is a simple calculation that can be performed quickly. It is used to calculate the output of a network for a given
    #  input, which is necessary for training and inference."
    
    outputs = net.forward(outputNames)          #output contains all the detections and it's confidence no.

    #print the output as a list as it is a list type element
    # print(outputs[0].shape)           
    # print(outputs[1].shape)           
    # print(outputs[2].shape)           

    #print(len(outputs))       this will give the length of output, which is 3 as output layers
    #print(type(output))       this will give the type of the output 
    #print(output)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#-------------------------------------------STAGE-7-----------------------------------------------------------------------------------------------------------------------------   
    findObjects(outputs,frame)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Now we open this output as an image to get the detection in the image

    cv.imshow("Video", frame)
    if cv.waitKey(1) & 0xff == ord('d'):
        break

cap.release()


