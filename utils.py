import time

import cv2
import face_recognition
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


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
    # Could also use (104.0, 117.0, 123.0) for mean, unclear which is the correct one
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    # [image_id, class, confidence, left, bottom, right, top]
    face_locations = net.forward()
    res = []
    for i in range(face_locations.shape[2]):
        confidence = face_locations[0, 0, i, 2]
        if confidence > 0.5:
            box = face_locations[0, 0, i, 3:7] * np.array([w, h, w, h])
            (left, bottom, right, top) = box.astype("int")
            res.append((top, right, bottom, left))

    return res


def load_haar_cascade(model_file):
    detector = cv2.CascadeClassifier()
    if not detector.load(model_file):
        print('--(!)Error loading face cascade')

    return detector


def load_cvdnn(model_file, config_file):
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    return net


def load_masknet(model_file):
    return load_model(model_file)


def get_centers(face_locations):
    return [(int((right + left)//2), int((bottom + top)//2)) for (top, right, bottom, left) in face_locations]


def detect_mask(frame, objects, clf):
    """Uses a pretrained model clf and extracted face locations
    to detect whether the person wears a mask. Classifier returns
    probabilities (mask, no_mask) for each provided face. The objects
    are (robustly) updated according to the classifier output.
    """

    # No faces detected
    if len(objects) == 0:
        return []

    face_locations = [o['bounding_box'] for o in objects.values()]
    # Extract the faces
    faces = [frame[bottom:top, left:right]
             for (top, right, bottom, left) in face_locations]
    # Preprocess the faces for the model
    faces = [img_to_array(cv2.resize(cv2.cvtColor(
        face, cv2.COLOR_BGR2RGB), (224, 224))) for face in faces]
    faces = preprocess_input(np.array(faces))
    masks = clf.predict(faces)

    # Update the mask information (Robust to intermittent false predictions)
    for o, (mask, no_mask) in zip(objects.values(), masks):
        if o['in_frame']:
            if o['has_mask']:
                if no_mask > mask:
                    o['consecutive_frames'] += 1
                else:
                    o['consecutive_frames'] = 0
            else:
                if mask > no_mask:
                    o['consecutive_frames'] += 1
                else:
                    o['consecutive_frames'] = 0
        if o['consecutive_frames'] > 3:
            o['has_mask'] = not o['has_mask']
            o['consecutive_frames'] = 0


def draw_bounding_boxes(frame, objects):
    for o_id, o in objects.items():
        if o['has_mask']:
            frame_color = (0, 255, 0)
        else:
            # Default color red
            frame_color = (0, 0, 255)

        cv2.circle(frame, o['centroid'], radius=1,
                   color=(0, 255, 0), thickness=-1)

        (top, right, bottom, left) = o['bounding_box']
        cv2.rectangle(frame, (left, top), (right, bottom), frame_color, 2)

        cv2.putText(frame, f"ID:{o_id}", (left, bottom-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, frame_color)

    return frame


def update_objects(objects, bounding_boxes, next_id):
    """Associates the newly detected objects with already known objects
    if it is possible. The pairs are determined by euclidean distance
    between centroids.
    """

    # How long to wait in seconds before forgetting an object that
    # is not in frame.
    time_to_forget = 3

    centroids = get_centers(bounding_boxes)

    pairs = {}
    object_ids = list(objects.keys())
    object_centroids = np.array([o['centroid'] for o in objects.values()])

    if len(object_centroids) != 0 and len(centroids) != 0:
        distances = cdist(object_centroids, centroids)

        nearest_centroids = distances.argmin(axis=1)
        # Loop through objects in closest neighbour order
        for i in distances.min(axis=1).argsort():
            if nearest_centroids[i] in pairs.values():
                continue

            objects[object_ids[i]
                    ]['bounding_box'] = bounding_boxes[nearest_centroids[i]]
            objects[object_ids[i]
                    ]['centroid'] = centroids[nearest_centroids[i]]
            objects[object_ids[i]]['last_detected_time'] = time.time()
            objects[object_ids[i]]['in_frame'] = True
            pairs[object_ids[i]] = nearest_centroids[i]

    # Forget the objects that haven't been seen for a while
    for o_id, val in list(objects.items()):
        if o_id not in pairs.keys():
            objects[o_id]['in_frame'] = False
            if time.time() - val['last_detected_time'] > time_to_forget:
                del objects[o_id]

    # Add the new objects
    for i, (cent, bb) in enumerate(zip(centroids, bounding_boxes)):
        if i not in pairs.values():
            objects[next(next_id)] = {'bounding_box': bb,
                                      'centroid': cent,
                                      'has_mask': True,
                                      'last_detected_time': time.time(),
                                      'consecutive_frames': 0,
                                      'in_frame': True
                                      }
