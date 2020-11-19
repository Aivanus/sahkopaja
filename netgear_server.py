#!/usr/bin/env python3

import argparse
import time

import cv2
from vidgear.gears import NetGear, PiGear


def main(args):

    # Start capturing live Monitor screen frames with default settings
    # time_delay for warming the camera up
    stream = PiGear(time_delay=2, rotation=args.rotation, resolution=args.resolution,
                    framerate=args.framerate, logging=args.logging).start()

    server_options = {'compression_format': '.jpg',
                      'compression_param': [cv2.IMWRITE_JPEG_QUALITY, args.compression_quality],
                      'flag': 1}
    server = NetGear(address=args.address, port=args.port, protocol='tcp', bidirectional_mode=True,
                     pattern=1, logging=False, **server_options)

    # loop over until KeyBoard Interrupted
    frame_counter = 0
    start_time = None
    while True:
        try:
            # read frames from stream
            frame = stream.read()
            if start_time is None:
                start_time = time.time()

            # check for frame if Nonetype
            if frame is None:
                break

            # send frame to server
            recv_data = server.send(frame)

            # print data just received from Client
            if not(recv_data is None):
                print(recv_data)
        except KeyboardInterrupt:
            break

        frame_counter += 1

    # safely close video stream
    stream.stop()

    # safely close server
    server.close()

    elapsed_time = time.time() - start_time
    print(f"sender avg FPS: {frame_counter/elapsed_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raspberry Pi server that sends video stream from PiCamera.")

    parser.add_argument('--address', type=str, default='192.168.0.103',
                        help="The IP adress of the client device, i.e. running this code.")
    parser.add_argument('--port', type=int, default=12345,
                        help="The port to listen to.")

    parser.add_argument('--rotation', type=int, default=0,
                        help="Rotate the camera output by given degrees.")
    parser.add_argument('--resolution', type=int, nargs=2, default=[320, 240],
                        help="The resolution of the camera captures.")
    parser.add_argument('--framerate', type=int, default=30,
                        help="The framerate at which the camera will capture.")
    parser.add_argument('--logging', action='store_true',
                        help="Allow logging messages.")

    parser.add_argument('--compression_quality', type=int, default=50,
                        help="The quality of the compressed image (0-100).")

    args = parser.parse_args()
    main(args)
