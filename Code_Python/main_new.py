import cv2
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

from keras.applications.resnet50 import ResNet50

isSocket = 0
debug    = 0

#______________________________________________________________

# import sys, termios, atexit
# from select import select
#
# # save the terminal settings
# fd = sys.stdin.fileno()
# new_term = termios.tcgetattr(fd)
# old_term = termios.tcgetattr(fd)
#
# # new terminal setting unbuffered
# new_term[3] = (new_term[3] & ~termios.ICANON & ~termios.ECHO)
#
# # switch to normal terminal
# def set_normal_term():
#     termios.tcsetattr(fd, termios.TCSAFLUSH, old_term)
#
# # switch to unbuffered terminal
# def set_curses_term():
#     termios.tcsetattr(fd, termios.TCSAFLUSH, new_term)
#
# def putch(ch):
#     sys.stdout.write(ch)
#
# def getch():
#     return sys.stdin.read(1)
#
# def getche():
#     ch = getch()
#     putch(ch)
#     return ch
#
# def kbhit():
#     dr,dw,de = select([sys.stdin], [], [], 0)
#     return dr
#
# def keyPressStart():
#     atexit.register(set_normal_term)
#     set_curses_term()
#
# def keyPressEnd():
#     atexit.register(set_curses_term)
#     set_normal_term()

#______________________________________________________

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

while True:

    print('Start server? (y for yes; m for yes WITH mood; any other to stop)')
    inx = input()
    if(inx!='y' and inx!='m'):
        break

    # print('Server Started. Press `q` to stop, `p` to pause. (Keys to be pressed on terminal)')
    # keyPressStart()

    frame_w = 1200
    border_w = 2
    min_size_w = 240
    min_size_h = 240
    min_size_w_eye = 60
    min_size_h_eye = 60
    scale_factor = 1.1
    min_neighbours = 5
    Engage = list()
    emoAccuracy = list()


    video_capture = cv2.VideoCapture(0)

    cntTime = 0
    cntFear = 0
    cntAnger = 0
    cntDisgust = 0
    cntNeutral = 0
    cntSadness = 0
    cntSurprise = 0
    cntHappiness = 0
    cntFocused = 0
    cntDistracted = 0

    while True:


        if not video_capture.isOpened():
            print('Unable to load camera.')
        else:
            ret, frame = video_capture.read()
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
                msg = 'No Face Detected;-;Distracted;Confidence= 100%;y;' #5th arguement is 'y' ie 'yes, pause the video'
                msg += str(cntTime)
                Engage.append(0)

            else:
                #increse time
                cntTime += 1
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
                emoAccuracy.append(prediction)
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

                probs_mean = np.mean(probs)
                if(np.isnan(probs_mean)):
                    Engage.append(np.random.uniform(0.6,0.7))
                    distraction_acc = 1 - Engage[-1]
                else:
                    Engage.append(probs_mean)
                    distraction_acc = 1 - Engage[-1]

                # get label
                if probs_mean <= 0.5:
                    label = 'Distracted'
                    pause = 'y'
                    cntDistracted += 1
                else:
                    label = 'Focused'
                    pause = 'n'
                    cntFocused += 1

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


                text = top + ' + ' + label
                cv2.putText(frame, text, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

                msg = ';'.join([emotion,str(emotion_acc),distraction,str(distraction_acc),pause,str(cntTime)])
                disp = ' '.join([emotion,str(emotion_acc),distraction,str(distraction_acc),'Frame#',str(cntTime)])

            if debug:
                print(disp)
            if(inx=='m'):
                cv2.imwrite('../public_static/frames/fig'+str(cntTime%10)+'.png', frame)

            if isSocket==1:
                socket.emit('emoNode', msg)

            if isSocket==0:
                cv2.imshow('Video', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # if kbhit():
            #     ch = getch()
            #     if(ch=='p'):
            #         keyPressEnd()
            #         video_capture.release()
            #         cv2.destroyAllWindows()
            #
            #         if isSocket==1:
            #             socket.emit('emoNode', 'pause;pause;pause;pause;pause;pause')
            #
            #
            #         print('Server paused. Press any key and then press enter to resume.')
            #         x=input()
            #         video_capture = cv2.VideoCapture(0)
            #         keyPressStart()
            #         print('Server Resumed. Press `q` to stop, `p` to pause. (Keys to be pressed on terminal)')
            #     if(ch=='q'):
            #         break


    if isSocket==1:
        socket.emit('emoNode', 'saving;saving;saving;saving;saving;saving')



    #Engagement Level Graph
    time_array = np.arange(cntTime + 1)

    emoAccuracy = np.array(emoAccuracy)
    emoAccuracy = np.reshape(emoAccuracy,(-1,7))
    print(emoAccuracy.shape)

    trace1 = go.Scatter(
            x = time_array,
            y = np.asarray(Engage)
            )
    data1=go.Data([trace1])
    layout1=go.Layout(title="Engagement Level Analysis", xaxis={'title':'Video Time'}, yaxis={'title':'Engagement Level'})
    figure1=go.Figure(data=data1,layout=layout1)
    pio.write_image(figure1, '../public_static/analytics/fig1.png')
    #pio.write_image(figure1, 'fig1.png')

    #Mood Pie Chart

    labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']
    values = [cntAnger, cntDisgust, cntFear, cntHappiness, cntSadness, cntSurprise]
    trace2 = go.Pie(labels = labels,values = values)
    data2=go.Data([trace2])
    layout2=go.Layout(title="Mood Distribution")
    figure2=go.Figure(data=data2,layout=layout2)
    pio.write_image(figure2, '../public_static/analytics/fig2.png')
    #pio.write_image(figure2, 'fig2.png')

    #Focused Distracted Bar graph
    trace3 = go.Pie(
        labels = ['Focused','Distracted'],
        values = [cntFocused,cntDistracted]
    )
    layout3 = go.Layout(title = "Focus Distribution")
    data3 = go.Data([trace3])
    figure3 = go.Figure(data = data3, layout = layout3)
    print("\nsaving image...")
    pio.write_image(figure3, '../public_static/analytics/fig3.png')
    print("saved image\n")
    traces = list()

    traces.append(go.Scatter(
        x = time_array,
        y = emoAccuracy[:,0],
        line = dict(color = 'red'),
        mode = 'lines',
        name = 'Anger'
    ))
    traces.append(go.Scatter(
        x = time_array,
        y = emoAccuracy[:,1],
        line = dict(color = 'yellow'),
        mode = 'lines',
        name = 'Disgust'
    ))
    traces.append(go.Scatter(
        x = time_array,
        y = emoAccuracy[:,2],
        line = dict(color = 'green'),
        mode = 'lines',
        name = 'Fear'
    ))
    traces.append(go.Scatter(
        x = time_array,
        y = emoAccuracy[:,3],
        line = dict(color = 'blue'),
        mode = 'lines',
        name = 'Happiness'
    ))
    traces.append(go.Scatter(
        x = time_array,
        y = emoAccuracy[:,4],
        line = dict(color = 'red'),
        mode = 'lines',
        name = 'Sadness'
    ))
    traces.append(go.Scatter(
        x = time_array,
        y = emoAccuracy[:,5],
        line = dict(color = 'yellow'),
        mode = 'lines',
        name = 'Surprise'
    ))
    traces.append(go.Scatter(
        x = time_array,
        y = emoAccuracy[:,6],
        line = dict(color = 'blue'),
        mode = 'lines',
        name = 'Neutral'
    ))

    for i in range(7):

        data4 = go.Data([traces[i]])
        layout4 = go.Layout(
            title = emotions[i] + ' Confidence v Video Time',
        )
        figure4 = go.Figure(data = data4, layout = layout4)
        datapath = '../public_static/analytics/'
        print(datapath + 'emofig' + str(i) + '.png')
        pio.write_image(figure4, datapath + 'emofig' + str(i) + '.png')

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
    #keyPressEnd()
    print('Figures saved')
    if isSocket==1:
        socket.emit('emoNode', 'end;end;end;end;end;end')


print('Server Ended')
