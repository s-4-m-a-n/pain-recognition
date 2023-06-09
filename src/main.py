
from tensorflow import keras
import cv2
import numpy as np
#import dlib


capture = cv2.VideoCapture("../test-videos/test_1.mp4")
cv2.namedWindow("Display", cv2.WINDOW_NORMAL)

frameSize = (500, 500)
out = cv2.VideoWriter("output_2.avi", cv2.VideoWriter_fourcc('M','J','P','G'),
                          10, frameSize)
save = True
IMG_SHAPE = 224

model = keras.models.load_model("../model/vggface_e50_acc_93_Va97.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

MAPPER = {0:"in pain" , 1: "not in pain"}

while True:
    is_true, frame = capture.read()

    # grayscaling
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #face detection
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    
    if len(faces):            
        x, y, w, h = faces[0]
        face_cropped = frame[y: y+h, x: x+w]
 

        img_data = cv2.resize(face_cropped, (IMG_SHAPE, IMG_SHAPE))
        prediction = model.predict(img_data.reshape(-1, IMG_SHAPE, IMG_SHAPE, 3))
        detection = np.argmax(prediction, axis=1)[0]
        # score = prediction[detection]


        frame = cv2.rectangle(frame, (x,y), (x+w, y+w), (255, 0, 0), 1)
        cv2.putText(frame, "status: {}".format(MAPPER[detection]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    cv2.imshow('Display', frame)

    if save:
        resized = cv2.resize(frame, (500,500), interpolation = cv2.INTER_AREA)
        out.write(resized)

    if cv2.waitKey(20) & 0xFF == ord('d'):
        break


capture.release()
out.release()
cv2.destroyAllWindows()
