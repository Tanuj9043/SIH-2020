import cv2
import urllib.request
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing.image import img_to_array
import imutils
import plotly as py
import plotly.graph_objs as go
import plotly.io as pio
import socket
import time
from socketIO_client import SocketIO, LoggingNamespace

url='http://192.168.0.27:6000/shot.jpg'
isSocket = 1

if isSocket==1:
    socket =  SocketIO('localhost', 8080, LoggingNamespace)
    print('Connection started')

# Model used
train_model =  "ResNet"
img_width, img_height = 197, 197
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']


model = load_model('models/ResNet-50.h5')
faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')
distract_model = load_model('models/distraction_model.hdf5', compile=False)


frame_w = 1200
border_w = 2
min_size_w = 240
min_size_h = 240
min_size_w_eye = 60
min_size_h_eye = 60
scale_factor = 1.1
min_neighbours = 5
Engage = list()

#video_capture = cv2.VideoCapture(0)

def preprocess_input(image):
    image = cv2.resize(image, (img_width, img_height))  # Resizing images for the trained model
    ret = np.empty((img_height, img_width, 3))
    ret[:, :, 0] = image
    ret[:, :, 1] = image
    ret[:, :, 2] = image
    x = np.expand_dims(ret, axis = 0)
    x -= 128.8006   # np.mean(train_dataset)
    x /= 64.6497    # np.std(train_dataset)
    return x

cntTime = 0
cntFear = 0
cntAnger = 0
cntDisgust = 0
cntNeutral = 0
cntSadness = 0
cntSurprise = 0
cntHappiness = 0

while True:
    #increse time
    cntTime += 1

    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgNp,-1)

    frame = imutils.resize(frame, width=frame_w)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detects objects of different sizes in the input image. The detected objects are returned as a list of rectangles
        # image:		Matrix of the type CV_8U containing an image where objects are detected
        # scaleFactor:	Parameter specifying how much the image size is reduced at each image scale
        # minNeighbors:	Parameter specifying how many neighbors each candidate rectangle should have to retain it
        # minSize:		Minimum possible object size. Objects smaller than that are ignored

    faces = faceCascade.detectMultiScale(
        gray_frame,
        scaleFactor		= scale_factor,
        minNeighbors	= min_neighbours,
        minSize			= (min_size_h, min_size_w))

    prediction = None
    x, y = None, None

    if len(faces) ==0 :
        msg = 'No Face Detected;NaN;Distracted;Confidence= 100%;y' #5th arguement is 'y' ie 'yes, pause the video' 
        Engage.append(0)

    else:
        biggest_face_index = 0
        biggest_face_size = 0
        ixx = 0
        for (x, y, w, h) in faces:
            if w*h > biggest_face_size:
                biggest_face_size = w*h
                biggest_face_index = ixx
            ixx += 1
            
        (x, y, w, h) = faces[biggest_face_index]
        ROI_gray = gray_frame[y:y+h, x:x+w]
        ROI_color = frame[y:y+h, x:x+w]
        # Draws a simple, thick, or filled up-right rectangle
            # img:          Image
            # pt1:          Vertex of the rectangle
            # pt2:          Vertex of the rectangle opposite to pt1
            # rec:          Alternative specification of the drawn rectangle
            # color:        Rectangle color or brightness (BGR)
            # thickness:    Thickness of lines that make up the rectangle. Negative values, like CV_FILLED ,
            #               mean that the function has to draw a filled rectangle
            # lineType:     Type of the line
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        emotion = preprocess_input(ROI_gray)
        prediction = model.predict(emotion)
        #print(emotions[np.argmax(prediction)] + " predicted with accuracy " + str(max(prediction[0])))
        top = emotions[np.argmax(prediction)]
        emotion = top
        emotion_acc = int(max(prediction[0])*100)
        emotion_acc = 'Confidence= ' + str(emotion_acc) + '%'

        eyes = eye_cascade.detectMultiScale(ROI_gray, scaleFactor=scale_factor,minNeighbors=min_neighbours,minSize=(min_size_w_eye,min_size_w_eye))


        probs = list()

        # loop through detected eyes
        for (ex,ey,ew,eh) in eyes:
            # draw eye rectangles
            cv2.rectangle(ROI_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),border_w)
            # get colour eye for distraction detection
            roi = ROI_color[ey+border_w:ey+eh-border_w, ex+border_w:ex+ew-border_w]
            # match CNN input shape
            roi = cv2.resize(roi, (64, 64))
            # normalize (as done in model training)
            roi = roi.astype("float") / 255.0
            # change to array
            roi = img_to_array(roi)
            # correct shape
            roi = np.expand_dims(roi, axis=0)

            # distraction classification/detection
            pred = distract_model.predict(roi)
            # save eye result
            probs.append(pred[0])

        # get average score for all eyes
        probs_mean = np.mean(probs)

        if(np.isnan(probs_mean)):
            probs_mean = (np.random.uniform(0.6,0.7))

        # get label
        if probs_mean <= 0.5:
            label = 'distracted'
            distraction_acc = 1 - probs_mean
            pause = 'y'
        else:
            label = 'focused'
            distraction_acc = probs_mean
            pause = 'n'

        distraction = label
        distraction_acc = int(distraction_acc*100)
        distraction_acc = 'Confidence= ' + str(distraction_acc) + '%'
        
        #increase cnt
        if top == 'Anger':
            cntAnger += 1
        if top == 'Disgust':
            cntDisgust += 1
        if top == 'Fear':
            cntFear += 1
        if top == 'Happiness':
            cntHappiness += 1
        if top == 'Sadness':
            cntSadness += 1
        if top == 'Surprise':
            cntSurprise += 1
        if top == 'Neutral':
            cntNeutral += 1

        #Append engagement level
        Engage.append(probs_mean)

        text = top + ' + ' + label
        cv2.putText(frame, text, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

        msg = ';'.join([emotion,str(emotion_acc),distraction,str(distraction_acc),pause]) 

    print(msg)
    if isSocket==1:
        socket.emit('emoNode', msg)

    if isSocket==0:
        cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#Engagement Level Graph
time = np.arange(cntTime + 1)
engm = np.asarray(Engage)
trace1 = go.Scatter(
        x = time,
        y = engm
        )
data1=go.Data([trace1])
layout1=go.Layout(title="Engagement Level Analysis", xaxis={'title':'Video Time'}, yaxis={'title':'Engagement Level'})
figure1=go.Figure(data=data1,layout=layout1)
pio.write_image(figure1, 'fig1.png')

#Mood Pie Chart

labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
values = [cntAnger, cntDisgust, cntFear, cntHappiness, cntSadness, cntSurprise, cntNeutral]
trace2 = go.Pie(labels = labels,values = values)
data2=go.Data([trace2])
layout2=go.Layout(title="Mood Distribution")
figure2=go.Figure(data=data2,layout=layout2)
pio.write_image(figure2, 'fig2.png')

# When everything is done, release the capture
#video_capture.release()
cv2.destroyAllWindows()
