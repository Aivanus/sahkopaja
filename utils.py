import cv2
import face_recognition
import numpy as np

# TODO: unify the bounding box coords

def get_face_locations_hog(frame):
    """
    Find faces from an image frame using HOG features from dlib (provided via
    face recognition package)
    """
    rgb_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    return face_locations


def get_face_locations_haar(frame, face_cascade):
    """Find faces from an image frame using Haar cascades from opencv.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    face_locations = face_cascade.detectMultiScale(frame_gray)
    face_locations = [(x, y, x+w, y+h) for (x, y, w, h) in face_locations]
    return face_locations


def get_face_locations_dnn(frame, net):
    """
    Find faces from an image frame using pretrained deep neural network
    from opencv.
    """
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

# TODO: make loading more flexible
def load_haar_cascade(model_file):
    # model_file='haarcascade_frontalface_alt.xml'
    detector = cv2.CascadeClassifier()
    if not detector.load(model_file):
        print('--(!)Error loading face cascade')

    return detector

def load_cvdnn(model_file, config_file):
    # model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    # config_file = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net

# TODO: make drawing bbs more general
def draw_bounding_boxes(frame, face_locations):
    for (top, right, bottom, left) in face_locations:
        center_x = (right + left)//2
        center_y = (bottom + top)//2
        cv2.circle(frame, (center_x, center_y), radius=1,
                   color=(0, 255, 0), thickness=-1)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    return frame

def get_coordinates():
    raise NotImplementedError("Function not ready yet!")
