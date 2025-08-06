#!/usr/bin/env -S uv run --script

import time
import logging
import sys
import os

import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import MultiprocessEmioCamera
from emioapi._logging_config import logger


def main():
    emio = MultiprocessEmioCamera()

    emio.show_frames = True # Show camera frames. Default is False.
    emio.track_markers = True # Track objects. Default is True.
    emio.compute_point_cloud = True # Compute point cloud. Default is False.

    # Camera tracking parameters. If None, a config file is read. If there is no config file, default values are used {"hue_h": 90, "hue_l": 36, "sat_h": 255, "sat_l": 138, "value_h": 255, "value_l": 35, "erosion_size": 0, "area": 100}
    emio.parameters = None 

    try:
        logger.info("List of available cameras: "+str(MultiprocessEmioCamera.listCameras()))

        if emio.open(): # Open the camera with the current parameters. This will start the camera process.
            logger.info(f"Camera {emio.camera_serial} opened successfully.")

            i = 0
            while i<20:
                time.sleep(1)
                print("--" * 20)
                
                # all properties of the emiocamera
                logger.info(f"Camera parameters: {emio.parameters}")
                logger.info(f"Camera show: {emio.show_frames}")
                logger.info(f"Camera tracking: {emio.track_markers}")
                logger.info(f"Camera compute point cloud: {emio.compute_point_cloud}")
                logger.info(f"Camera is running: {emio.is_running}")
                logger.info(f"Count tracker positions: {len(emio.trackers_pos)}")
                logger.info(f"Point cloud shape: {emio.point_cloud.shape}")
                logger.info(f"HSV Frame shape: {emio.hsv_frame.shape}")
                logger.info(f"Mask Frame shape: {emio.mask_frame.shape}")

                i += 1

            emio.close()


    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Closing camera...")
        emio.close()
        return
    except Exception as e:
        logger.exception(f"Failed running camera: {e}")
        emio.close()
        return


if __name__ == "__main__":
    try:
        logger.info("Starting EMIO API test...")
        main()
        logger.info("EMIO API test completed successfully.")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")