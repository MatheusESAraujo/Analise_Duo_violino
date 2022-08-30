import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns 
import glob, os , datetime
import imutils
from scipy.spatial import distance
import cv2
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")



def aplica_rede(frame,motion,previous_frame,param_lim):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
            
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
 
        # draw the bounding box of the face along with the associated
        # probability
    text = "{:.2f}%".format(detections[0, 0, 0, 2] * 100)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
    centroid = (sum([startX, endX]) / 2, sum([startY, endY]) / 2)
    
    if centroid[0]>0 and centroid[1]>0:
        cv2.circle(frame, (int(centroid[0]),int(centroid[1])), radius=10, color=(0, 0, 255), thickness=-1)
        if len(motion)>1:

            limite = int((1+param_lim)*motion[:-1][0])
            #limite_inf = int((1-0.15)*motion[:-1][0])
        else:
            limite=350
            #limite_inf = 0
        
        if distance.euclidean(previous_frame, centroid)<limite: #and distance.euclidean(previous_frame, centroid)>limite_inf :
            motion.append(distance.euclidean(previous_frame, centroid))
            previous_frame = centroid
        # elif distance.euclidean(previous_frame, centroid)<limite:
        #     motion.append(limite_inf)
        #     previous_frame = centroid

        else:
            motion.append(limite)
            previous_frame = centroid
    else:
        motion.append(distance.euclidean(previous_frame, previous_frame))
        previous_frame = previous_frame
        
        
        
    cv2.putText(frame, text, (startX, y),
    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    return frame,motion,previous_frame




def analise_camera(camera=0,tamanho=800,param_lim=0.15):
    cap = cv2.VideoCapture(camera)




    motion_right=[]
    motion_left=[]
    previousframe_right=(0,0)
    previousframe_left=(0,0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=tamanho)
        height, width, ch = frame.shape

        roi_height = height 
        roi_width = int(tamanho/2)
        frame[0:roi_height, 0:roi_width]
        
        images = [frame[0:roi_height, 0:roi_width],frame[0:roi_height, roi_width:2*roi_width]]
        
        imagem_right,motion_right,previousframe = aplica_rede(images[0],motion_right,previousframe_right,param_lim)
        imagem_left,motion_left,previousframe = aplica_rede(images[1],motion_left,previousframe_left,param_lim)

        cv2.imshow("Frame - Right",imagem_right )
        plt.clf()
        plt.plot(motion_right)
        plt.savefig('temp_right.png')
        img1 = cv2.imread('temp_right.png')
        cv2.moveWindow("Frame - Right", roi_width-int(tamanho/4),0)
        cv2.imshow("Plot1",imutils.resize(img1, width=300))
        cv2.moveWindow("Plot1", 250,500)
        cv2.imshow("Frame - Left", imagem_left)
        plt.clf()
        plt.plot(motion_left)
        plt.savefig('temp_left.png')
        img2 = cv2.imread('temp_left.png')
        cv2.imshow("Plot2",imutils.resize(img2, width=300))
        cv2.moveWindow("Plot2", 650,500)
        
        cv2.moveWindow("Frame - Left", roi_width+int(tamanho/4),0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()





def analise_video_imshow(local_video,tamanho=800,param_lim=0.15):
    cap = cv2.VideoCapture(local_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )

    motion_right=[]
    motion_left=[]
    previousframe_right=(0,0)
    previousframe_left=(0,0)
   
    for i in tqdm(range(length)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=tamanho)
        height, width, ch = frame.shape

        roi_height = height 
        roi_width = int(tamanho/2)
        frame[0:roi_height, 0:roi_width]
        
        images = [frame[0:roi_height, 0:roi_width],frame[0:roi_height, roi_width:2*roi_width]]
        
        imagem_right,motion_right,previousframe = aplica_rede(images[0],motion_right,previousframe_right,param_lim)
        imagem_left,motion_left,previousframe = aplica_rede(images[1],motion_left,previousframe_left,param_lim)

        cv2.imshow("Frame - Right",imagem_right )
        plt.clf()
        plt.plot(motion_right)
        plt.savefig('temp_right.png')
        img1 = cv2.imread('temp_right.png')
        cv2.moveWindow("Frame - Right", roi_width-int(tamanho/4),0)
        cv2.imshow("Plot1",imutils.resize(img1, width=300))
        cv2.moveWindow("Plot1", 250,500)
        cv2.imshow("Frame - Left", imagem_left)
        plt.clf()
        plt.plot(motion_left)
        plt.savefig('temp_left.png')
        img2 = cv2.imread('temp_left.png')
        cv2.imshow("Plot2",imutils.resize(img2, width=300))
        cv2.moveWindow("Plot2", 650,500)
        
        cv2.moveWindow("Frame - Left", roi_width+int(tamanho/4),0)



        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame()
    df["motion_right"] = motion_right
    df["motion_left"] = motion_left
    list1 = [i for i in df['motion_right'].values]
    list2 = [i for i in df['motion_left'].values]
    lista_fim = list1 + list2
    scaler = MinMaxScaler().fit(pd.DataFrame(lista_fim))

    df["motion_right_scaled"] = scaler.transform(df[['motion_right']])
    df["motion_left_scaled"] = scaler.transform(df[['motion_left']])



    return df




def analise_video(local_video,tamanho=800,param_lim=0.15):
    cap = cv2.VideoCapture(local_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( length )

    motion_right=[]
    motion_left=[]
    previousframe_right=(0,0)
    previousframe_left=(0,0)
   
    for i in tqdm(range(length)):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=tamanho)
        height, width, ch = frame.shape

        roi_height = height 
        roi_width = int(tamanho/2)
        frame[0:roi_height, 0:roi_width]
        
        images = [frame[0:roi_height, 0:roi_width],frame[0:roi_height, roi_width:2*roi_width]]
        
        imagem_right,motion_right,previousframe = aplica_rede(images[0],motion_right,previousframe_right,param_lim)
        imagem_left,motion_left,previousframe = aplica_rede(images[1],motion_left,previousframe_left,param_lim)




        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame()
    df["motion_right"] = motion_right
    df["motion_left"] = motion_left
    scaler = MinMaxScaler().fit(pd.DataFrame(df[['motion_right']]))

    df["motion_right_scaled"] = scaler.transform(df[['motion_right']])
    scaler = MinMaxScaler().fit(pd.DataFrame(df[['motion_left']]))
    df["motion_left_scaled"] = scaler.transform(df[['motion_left']])
    df["time"] = (df.index+1)/len(df)



    right = df.motion_right_scaled
    left = df.motion_left_scaled
    def trata_ruido_right(df):
        if df.motion_right_scaled >= right.max():
            return right.median()
        elif df.motion_right_scaled <= right.min():
            return right.median()
        else:
            return df.motion_right_scaled 
        
    def trata_ruido_left(df):
        if df.motion_left_scaled >= left.max():
            return left.median()
        elif df.motion_left_scaled <= left.min():
            return left.median()
        else:
            return df.motion_left_scaled 
        
    df["motion_right_scaled_trat"] = df.apply(trata_ruido_right,axis=1)
    df["motion_left_scaled_trat"] = df.apply(trata_ruido_left,axis=1)
    


    return df