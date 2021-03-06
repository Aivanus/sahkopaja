#!/usr/bin/env python3
import argparse
import time
from itertools import count

import cv2
from vidgear.gears import NetGear
import numpy as np

import utils


def main(args):
    client_options = {'compression_format': '.jpg',
                      'compression_param': cv2.IMREAD_COLOR,
                      'flag': 1}
    client = NetGear(address=args.address, port=args.port, protocol='tcp', bidirectional_mode=True,
                     pattern=1, receive_mode=True, logging=True, **client_options)

    face_detector = None
    mask_detector = None

    if args.detection_method == 'dnn':
        model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = "models/deploy.prototxt"
        face_detector = utils.load_cvdnn(model_file, config_file)
    elif args.detection_method == 'haar':
        model_file = 'models/haarcascade_frontalface_alt.xml'
        face_detector = utils.load_haar_cascade(model_file)

    if not args.ignore_masks:
        # Load mask detector
        # Credit for the model: https://github.com/chandrikadeb7/Face-Mask-Detection
        mask_detector = utils.load_masknet("models/mask_detector.model")

    # Keep track of FPS
    frame_counter = 0
    start_time = None

    # Bookkeeping for face tracking
    next_id = count(0)
    objects = {}

    target_coords = []
    target_bb_size = 0
    target = None
    while True:

        # receive frames from network
        data = client.recv(return_data=(target_coords, target_bb_size))
        if start_time is None:
            start_time = time.time()

        # check for received frame if Nonetype
        if data is None:
            break
        _, frame = data

        if frame_counter % args.frameskip == 0:
            if args.detection_method == 'dnn':
                face_locations = utils.get_face_locations_dnn(
                    frame, face_detector)
            else:
                face_locations = utils.get_face_locations_hog(frame)

            # Update the object positions and other parametes
            utils.update_objects(objects, face_locations, next_id)

            if frame_counter % args.mask_detect_freq == 0:
                # Update the mask status of the faces
                utils.detect_mask(frame, objects, mask_detector)

            # Choose the target
            if target not in objects.keys() or objects[target]['has_mask']:
                temp = [i for i, val in objects.items() if not val['has_mask']]
                if temp:
                    target = temp[0]
                else:
                    target = None
            if target is not None:
                target_coords = objects[target]['centroid']
                # Change to centered coordinates
                h, w, _ = frame.shape
                x0, y0 = (w//2, h//2)
                target_coords = (target_coords[0]-x0, target_coords[1]-y0)

                # Bounding box size
                (top, right, bottom, left) = objects[target]['bounding_box']
                target_bb_size = round(np.linalg.norm([(top, right), (bottom, left)]),2)

            else:
                target_coords = []
                target_bb_size = 0

        if args.show_video:
            # Draw the extra information onto the frame
            frame = utils.draw_bounding_boxes(frame, objects)
            # h, w, _ = frame.shape
            # cv2.circle(frame, (w//2, h//2), radius=5,
            #        color=(0, 255, 0), thickness=-1)

            # Show output
            if args.track_face and target is not None:
                (top, right, bottom, left) = objects[target]['bounding_box']
                cv2.imshow("Output Frame", frame[bottom:top, left:right])
            else:
                cv2.imshow("Output Frame", frame)

            # check for 'q' key if pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            frame_counter += 1

    elapsed_time = time.time() - start_time
    print(f"avg FPS: {frame_counter/elapsed_time}")

    # close output window
    cv2.destroyAllWindows()

    # safely close client
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect faces and extract their bounding boxes from a video stream.")

    parser.add_argument('--address', type=str, default='192.168.0.103',
                        help="The IP adress of the client device, i.e. running this code.")
    parser.add_argument('--port', type=int, default=12345,
                        help="The port to listen to.")

    parser.add_argument('--detection_method', type=str, choices=['hog', 'dnn', 'haar'],
                        default='dnn', help="Method used for face detection.")
    parser.add_argument('--frameskip', type=int, default=1,
                        help="Process every nth frame.")
    parser.add_argument('--mask_detect_freq', type=int, default=5,
                        help="How often to update mask information.")

    parser.add_argument('--show_video', action='store_false',
                        help="Show the processed video stream")
    parser.add_argument('--track_face', action='store_true',
                        help="Show a cropped face in the video stream.")
    parser.add_argument('--ignore_masks', action='store_true',
                        help="Do not detect masks.")

    args = parser.parse_args()
    main(args)
