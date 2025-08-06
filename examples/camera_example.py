#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioCamera
from emioapi._logging_config import logger


def main(emio: EmioCamera):

    emio.calibrate()  # calibrate the camera if needed

    while emio.is_running:
        try:
            emio.update() # update the camera frame and trackers

            print("-"*20)
            logger.info(f"Camera parameters: {emio.parameters}")
            logger.info(f"Camera show: {emio.show_frames}")
            logger.info(f"Camera tracking: {emio.track_markers}")
            logger.info(f"Camera compute point cloud: {emio.compute_point_cloud}")
            logger.info(f"Camera is running: {emio.is_running}")
            logger.info(f"Count tracker: {len(emio.trackers_pos)}")
            logger.info(f"Trackers positions: {emio.trackers_pos}")
            logger.info(f"Point cloud shape: {emio.point_cloud.shape}")
            logger.info(f"HSV Frame shape: {emio.hsv_frame.shape}")
            logger.info(f"Mask Frame shape: {emio.mask_frame.shape}")

        except KeyboardInterrupt: 
            logger.info("Keyboard interrupt received.")
            break
        except Exception as e:
            logger.exception(f"Error during communication: {e}")
            break


if __name__ == "__main__":
    try:
        logger.info("Starting EMIO Camera test...")
        
        logger.info("List of available cameras\n"+str(EmioCamera.listCameras()))

        logger.info("Opening and configuring EMIO Camera...")

        emio = EmioCamera(show=True, track_markers=True, compute_point_cloud=True)
        
        if emio.open(): # This will open the first available Realsense camera
            
            logger.info(f"Emio camera {emio.camera_serial} opened.")
            logger.info("Running main function...")
            main(emio)

            logger.info("Main function completed.")
            logger.info("Closing Emio API...")

            emio.close()

            logger.info("EMIO API closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")