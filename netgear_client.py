#!/usr/bin/env python3
import argparse
import time

import cv2
from vidgear.gears import NetGear

import utils


def main(args):
    client_options = {'compression_format': '.jpg',
                      'compression_param': cv2.IMREAD_COLOR}
    client = NetGear(address=args.address, port=args.port, protocol='tcp', bidirectional_mode=True,
                     pattern=1, receive_mode=True, logging=True, **client_options)

    detector = None
    if args.detection_method == 'dnn':
        model_file = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        config_file = "models/deploy.prototxt"
        detector = utils.load_cvdnn(model_file, config_file)
    elif args.detection_method == 'haar':
        model_file = 'models/haarcascade_frontalface_alt.xml'
        detector = utils.load_haar_cascade(model_file)

    # Keep track of FPS
    frame_counter = 0
    start_time = None

    coords = None
    face_locations = None
    while True:

        # receive frames from network
        data = client.recv(return_data=coords)
        if start_time is None:
            start_time = time.time()

        # check for received frame if Nonetype
        if data is None:
            break

        _, frame = data

        if frame_counter % args.frameskip == 0:
            if args.detection_method == 'dnn':
                face_locations = utils.get_face_locations_dnn(frame, detector)
            else:
                face_locations = utils.get_face_locations_hog(frame)

            coords = utils.get_centers(face_locations)

        frame = utils.draw_bounding_boxes(frame, face_locations)

        if args.show_video:
            # Show output window
            if args.track_face and face_locations:
                (top, right, bottom, left) = face_locations[0]
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
    parser.add_argument('--frameskip', type=int, default=1, help="Process every nth frame.")

    parser.add_argument('--show_video', action='store_false',
                        help="Show the processed video stream")
    parser.add_argument('--track_face', action='store_true',
                        help="Show a cropped face in the video stream.")

    args = parser.parse_args()
    main(args)
