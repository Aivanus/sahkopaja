import argparse
import os
import pickle
import time
from pathlib import Path

import cv2
import face_recognition
from face_recognition.api import face_locations
import imutils
import numpy as np
from imutils.video import FPS, VideoStream


def get_face_locations_hog(frame):
    rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    return face_locations


def get_face_locations_haar(frame, face_cascade):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    face_locations = face_cascade.detectMultiScale(frame_gray)
    face_locations = [(x, y, x+w, y+h) for (x, y, w, h) in face_locations]
    return face_locations


def get_face_locations_cv_dnn(frame, net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    face_locations = net.forward()
    res = []
    for i in range(face_locations.shape[2]):
        confidence = face_locations[0, 0, i, 2]
        if confidence > 0.5:
            box = face_locations[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            # Notice the weird permutation
            tt = (y, x1, y1, x)
            res.append(tt)

    return res


def load_cvdnn():
    modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

def draw_bounding_boxes(frame, face_locations):
    for (top, right, bottom, left) in face_locations:
            center_x = (right + left)//2
            center_y = (bottom + top)//2
            cv2.circle(frame, (center_x, center_y), radius=1,
                       color=(0, 255, 0), thickness=-1)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    return frame


def main(args):
    # Init the video stream and let the camera to warm up
    print("Lights!")
    vs = VideoStream(src=0, usePiCamera=True, resolution=(
        args.resolution[0], args.resolution[1]), framerate=args.framerate).start()
    print("Camera!")
    time.sleep(2.0)

    detector = None
    if args.detection_method == 'haar':
        haar_path = 'haarcascade_frontalface_alt.xml'
        detector = cv2.CascadeClassifier()
        if not detector.load(haar_path):
            print('--(!)Error loading face cascade')
    if args.detection_method == 'cvdnn':
        detector = load_cvdnn()

    # start the FPS throughput estimator
    fps = FPS().start()
    print("Action!")
    while True:
        frame = vs.read()
        frame = imutils.rotate(frame, args.rotate)

        if args.detection_method == 'hog':
            face_locations = get_face_locations_hog(frame)
        elif args.detection_method == 'haar':
            face_locations = get_face_locations_haar(frame, detector)
        elif args.detection_method == 'cvdnn':
            face_locations = get_face_locations_cv_dnn(frame, detector)
        else:
            raise NotImplementedError('This detection method is not ready!')

        frame = draw_bounding_boxes(frame, face_locations)

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

    parser.add_argument('--detection_method', type=str, choices=['hog', 'haar', 'cvdnn'],
                        default='hog', help="Rotate the camera output by given degrees.")

    parser.add_argument('--hide_stream', action='store_true',
                        help="Hide the video stream")
    parser.add_argument('--print_coords', action='store_true',
                        help="Print the coordinates of bounding boxes")
    parser.add_argument('--rotate', type=int, default=0,
                        help="Rotate the camera output by given degrees.")
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 240],
                        help="The resolution of the camera captures.")
    parser.add_argument('--framerate', type=int, default=32,
                        help="The framerate at which the camera will capture.")
    args = parser.parse_args()
    main(args)
