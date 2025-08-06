#!/usr/bin/env -S uv run --script

import time
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/..')
from emioapi import EmioMotors
from emioapi._logging_config import logger

def main(emio: EmioMotors, loops=1):

    initial_pos_pulse = [0] * 4
    emio.max_velocity = [1000] * 4
    logger.info(f"Initial position in rad: {initial_pos_pulse}")
    emio.angles = initial_pos_pulse
    time.sleep(1)
    emio.printStatus()


    for i in range(loops):
        new_pos = [((2*3.14)*((i+1)%8)/8)] * 4
        print("-"*20)
        logger.info(f"new_pos {new_pos}")
        try:
            if emio.is_connected:
                emio.angles = new_pos
                time.sleep(1)
                emio.printStatus()
            else:
                emio.open()
        except Exception as e:
            logger.error(f"Error during communication: {e}")
            emio.close()
            emio.open()


if __name__ == "__main__":
    try:
        logger.info("Starting EMIO API test...")
        logger.info("Opening and configuring EMIO API...")
        
        emio_motors = EmioMotors()
        
        if emio_motors.open(): 
            
            emio_motors.printStatus()

            logger.info("Emio motors opened and configured.")
            logger.info("Running main function...")
            main(emio_motors, 15)

            logger.info("Main function completed.")
            logger.info("Closing Emio motor connection...")

            emio_motors.close()
            logger.info("Emio connection closed.")
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        emio_motors.close()