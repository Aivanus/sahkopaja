#!/usr/bin/env python3

from vidgear.gears.asyncio import NetGear_Async
from vidgear.gears import NetGear, PiGear
import cv2
import asyncio

# initialize Server
options = {"vflip": True, "hflip": True, "exposure_mode": "auto", "iso": 800,
           "exposure_compensation": 15, "awb_mode": "horizon"}
server = NetGear_Async(address='192.168.0.103',
                       enablePiCamera=True, framerate=30,
                       time_delay=2, logging=True, **options).launch()

if __name__ == '__main__':
    # set event loop
    asyncio.set_event_loop(server.loop)
    try:
        # run your main function task until it is complete
        server.loop.run_until_complete(server.task)
    except (KeyboardInterrupt, SystemExit):
        # wait for interrupts
        pass
    finally:
        # finally close the server
        server.close()
