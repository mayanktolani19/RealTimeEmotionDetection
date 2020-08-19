import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from keras.preprocessing import image




model = model_from_json(open("model.json", "r").read())
model.load_weights('model_weights.h5')
face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
cap=cv2.VideoCapture(0)

class VideoCamera():
    while True:
        ret,test_img=cap.read()
        if not ret:
            continue
        gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)
        for (x,y,w,h) in faces_detected:
            emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
            fc = gray_img[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            preds = model.predict(roi[np.newaxis, :, :, np.newaxis])
            pred = emotions[np.argmax(preds)]
            cv2.putText(test_img, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),2)
            
    
            #find max indexed array
            max_index = np.argmax(pred[0])
    
            
            predicted_emotion = emotions[max_index]
    
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ',resized_img)
    
    
    
        if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
            break

cap.release()
cv2.destroyAllWindows