import argparse
import os
import pickle
import time
from pathlib import Path

import cv2
import face_recognition
import imutils
import numpy as np
from imutils.video import FPS, VideoStream
from mtcnn import MTCNN


def get_face_locations_hog(frame):
        rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        return face_locations

def get_face_locations_haar(frame, face_cascade):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        face_locations = face_cascade.detectMultiScale(frame_gray)
        face_locations = [(x, y, x+w, y+h) for (x,y,w,h) in face_locations]
        return face_locations

def get_face_locations_mtcnn(frame, detector):
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = detector.detect_faces(frame)
    face_locations = [(x, y, x+w, y+h) for (x,y,w,h) in face_locations]
    return face_locations

def main(args):
    # Init the video stream and let the camera to warm up
    print("Lights!")
    vs = VideoStream(src=0, usePiCamera=True).start()
    print("Camera!")
    time.sleep(2.0)

    if args.detection_method == 'haar':
        haar_path = 'haarcascade_frontalface_alt.xml'
        face_cascade = cv2.CascadeClassifier()
        if not face_cascade.load(haar_path):
            print('--(!)Error loading face cascade')
    if args.detection_method == 'mtcnn':
        mtcnn_detector = MTCNN()

    # start the FPS throughput estimator
    fps = FPS().start()
    print("Action!")
    while True:
        frame = vs.read()
        frame = imutils.rotate(frame, args.rotate)

        if args.detection_method == 'hog':
            face_locations = get_face_locations_hog(frame)
        elif args.detection_method == 'haar':
            face_locations = get_face_locations_haar(frame, face_cascade)
        elif args.detection_method == 'mtcnn':
            face_locations = get_face_locations_mtcnn(frame, mtcnn_detector)
        else:
            raise NotImplementedError('This detection method is not ready!')

        for (top, right, bottom, left) in face_locations:
            center_x = (right + left)//2
            center_y = (bottom + top)//2
            cv2.circle(frame, (center_x, center_y), radius=1,
                       color=(0, 255, 0), thickness=-1)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            if args.print_coords:
                print(f"({center_x, center_y})")
                print(face_locations)

        if not args.hide_stream:
            cv2.imshow("Frame", imutils.resize(frame, width=800))
        fps.update()

        key = cv2.waitKey(1) & 0xFF
        # if the `^q` key was pressed, break from the loop
        if key == ord("q"):
            break

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect faces and extract their bounding boxes from a video stream.")

    parser.add_argument('--detection_method', type=str, choices=['hog', 'haar', 'mtcnn'],
                        default='hog', help="Rotate the camera output by given degrees.")

    parser.add_argument('--hide_stream', action='store_true',
                        help="Hide the video stream")
    parser.add_argument('--print_coords', action='store_true',
                        help="Print the coordinates of bounding boxes")
    parser.add_argument('--rotate', type=int, default=0,
                        help="Rotate the camera output by given degrees.")
    args = parser.parse_args()
    main(args)
