import cv2
import numpy as np
import dlib
import os
import random

RANDOM_TOKEN = random.getrandbits(128)

def video_to_dataset(video_path):
    prefix = str(RANDOM_TOKEN)+video_path.split(".")[0].split("/")[-1]
    print(prefix)

    capture = cv2.VideoCapture(video_path)
    cv2.namedWindow("Display", cv2.WINDOW_NORMAL)
    # Using resizeWindow()
    cv2.resizeWindow("Display", 500, 500)
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    DESTINATION_ROOT = "dataset/"
    c = 0
    IMG_SHAPE = 128
    
    while True:
        is_true, frame = capture.read()

        if not is_true:
            break
        # grayscaling
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #face detection
        faces = face_detector.detectMultiScale(gray_image, 1.1, 4)
        second_value = 0
        
        if len(faces):
            for face in faces:          
                x, y, w, h = face
                face_cropped = frame[y: y+h, x: x+w]
                try:
                    img_data = cv2.resize(face_cropped, (IMG_SHAPE, IMG_SHAPE))

                    cv2.imshow('Display', img_data)
                    key = cv2.waitKey(0) & 0xFF

                    if key == ord('a'):
                        cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "0"),
                                                 prefix+"_"+str(c)+".jpg"),
                                                 img_data)
                        c += 1
                    elif key == ord('b'):
                        cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "1"),
                                                 prefix+"_"+str(c)+".jpg"),
                                                 img_data)
                        c += 1

                    elif key == ord('c'):
                        cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "2"),
                                                 prefix+"_"+str(c)+".jpg"),
                                                 img_data)
                        c += 1
                    elif  key == ord('x'):
                        break

                except Exception as e:
                    print(e)
                    continue
        else:
            continue

        # cv2.imshow('Display', img_data)
        # key = cv2.waitKey(0) & 0xFF

        # if key == ord('a'):
        #     cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "0"),
        #                              prefix+"_"+str(c)+".jpg"),
        #                              img_data)
        #     c += 1
        # elif key == ord('b'):
        #     cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "1"),
        #                              prefix+"_"+str(c)+".jpg"),
        #                              img_data)
        #     c += 1

        # elif key == ord('c'):
        #     cv2.imwrite(os.path.join(os.path.join(DESTINATION_ROOT, "2"),
        #                              prefix+"_"+str(c)+".jpg"),
        #                              img_data)
        #     c += 1
        # elif  key == ord('x'):
        #     break

        #skipping next three frame
        for i in range(3):
            capture.read()

    cv2.destroyWindow('Display')
    capture.release()
    return -1

if __name__ == "__main__":
    video_to_dataset("videos/video_sample_34.mp4")