#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioAPI
from emioapi._logging_config import logger


"""
A simple example using both the camera and the motors.
Here new commands are sent every seconds to the motors.
Because we use a 1s sleep in the while loop, the camera won't be fluid because by default, it is launched in the same process as the while loo
"""

def main():
    emio = EmioAPI()

    # Camera     
    emio.camera.show_frames = True # Show camera frames. Default is False.
    emio.camera.track_markers = True # Track objects. Default is True.
    emio.camera.compute_point_cloud = True # Compute point cloud. Default is False.

    # Camera tracking parameters. If None, a config file is read. If there is no config file, default values are used {"hue_h": 90, "hue_l": 36, "sat_h": 255, "sat_l": 138, "value_h": 255, "value_l": 35, "erosion_size": 0, "area": 100}
    emio.camera.parameters = None

    try:
        logger.info("List of all Emio devices: "+str(EmioAPI.listEmioDevices()))
        if emio.connectToEmioDevice():  # Open the camera with the current parameters. This will start the camera process.
            logger.info("Successfully connected to Emio.")
            
            logger.info("List of used Emio devices: "+str(EmioAPI.listUsedEmioDevices())) # list all the used Emio devices
            logger.info("List of unused Emio devices: "+str(EmioAPI.listUnusedEmioDevices())) # list all the unused Emio devices

            #  Motors
            initial_pos_pulse = [0] * 4
            logger.info(f"Initial position in rad: {initial_pos_pulse}")
            emio.motors.max_velocity = [1000] * 4
            emio.motors.angles = initial_pos_pulse
            time.sleep(1)
            emio.printStatus()

            i = 0
            while i<10:
                print("--" * 20)
                time.sleep(1)
                # camera 
                emio.camera.update()
                logger.info(f"Camera parameters: {emio.camera.parameters}")
                logger.info(f"Camera show: {emio.camera.show_frames}")
                logger.info(f"Camera tracking: {emio.camera.track_markers}")
                logger.info(f"Camera compute point cloud: {emio.camera.compute_point_cloud}")
                logger.info(f"Camera is running: {emio.camera.is_running}")
                logger.info(f"Count tracker positions: {len(emio.camera.trackers_pos)}")
                logger.info(f"Point cloud shape: {emio.camera.point_cloud.shape}")
                logger.info(f"HSV Frame shape: {emio.camera.hsv_frame.shape}")
                logger.info(f"Mask Frame shape: {emio.camera.mask_frame.shape}")

                print("")

                #  Motors
                new_pos = [((2*3.14)*((i+1)%8)/8)] * 4

                if emio.motors.is_connected:
                    emio.motors.angles = new_pos
                    time.sleep(0.5)  # wait for the motors to move
                    logger.info(f"new_pos {new_pos}")
                    emio.motors.printStatus()

                i += 1
        
            emio.disconnect()


    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Closing camera...")
        emio.camera.close()
        return
    except Exception as e:
        logger.exception(f"Failed running camera: {e}")
        emio.disconnect()
        return


if __name__ == "__main__":
    try:
        logger.info("Starting EMIO API test...")
        main()
        logger.info("EMIO API example completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")