import os
import json
from time import sleep
import time
import tkinter as tk
from tkinter import ttk
from enum import Enum

import numpy as np
import cv2 as cv
import pyrealsense2 as rs

from ._camerafeedwindow import CameraFeedWindow
from ._positionestimation import PositionEstimation, image_pixel_to_mm
from emioapi._logging_config import logger

CONFIG_FILENAME = os.path.dirname(__file__) + '/cameraparameter.json'
DEFAULT_CAMERA_PARAMS = {"hue_h": 90, "hue_l": 36, "sat_h": 255, "sat_l": 138, "value_h": 255, "value_l": 35, "erosion_size": 0, "area": 100}

class CalibrationStatusEnum(Enum):
    NOT_CALIBRATED = 0,
    CALIBRATING = 1,
    CALIBRATED = 2


def compute_contour_center(contour):
    M = cv.moments(contour)
    cX = 0
    cY = 0
    if M['m00'] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cX, cY


def compute_median_depth(contour, color_image, depth_image):
    image = np.zeros_like(color_image)
    # Fills the area bounded by the contours if thickness < 0
    cv.drawContours(image, contours=[contour], contourIdx=0, color=255, thickness=-1)
    points = np.where(image == 255)
    depth_values = np.array([depth_image[p[0], p[1]] for p in points]).flatten()
    valid_depth_values = depth_values[depth_values > 0]
    if len(valid_depth_values) > 0:
        return np.median(valid_depth_values)
    else:
        return 0


def listCameras() -> list:
    context = rs.context()
    return [d.get_info(rs.camera_info.serial_number) for d in context.devices]


class DepthCamera:

    height = 480
    width = 640
    device = None
    pipeline_profile = None
    pipeline_wrapper = None
    config = None
    pipeline = None
    point_cloud = None
    intr = None
    profile = None
    initialized = False
    pc = None
    compute_point_cloud = False
    position_estimator: PositionEstimation = None
    parameter = {}
    tracking = True
    trackers_pos = []
    maskWindow = None
    frameWindow = None
    hsvWindow = None
    rootWindow = None
    hsvFrame = None
    maskFrame = None
    frame: np.ndarray = None
    depth_frame: np.ndarray = None
    calibration_status = CalibrationStatusEnum.NOT_CALIBRATED

    @property
    def camera_serial(self) -> str:
        """
        Returns the serial of the camera as str
        
        """
        return self.device.get_info(rs.camera_info.serial_number)


    def __init__(self, 
                 camera_serial: str=None, 
                 parameter: dict=None, 
                 compute_point_cloud: bool=False, 
                 show_video_feed: bool=False, 
                 tracking: bool=True,
                 configuration: str="extended") -> None:
        """
        Initialize the camera and the parameters.

        Args:
            parameter : dict
                The parameters for the camera. If None, the default parameters will be used.
            comp_point_cloud : bool
                If True, the point cloud will be computed.
            show_video_feed : bool
                If True, the video feed will be shown.
            track: bool
                If True, the tracking will be enabled.
            configuration: str
                Configuration of Emio, either "extended" (default) or "compact"
        """
        self.tracking = tracking
        self.show_video_feed = show_video_feed
        self.compute_point_cloud = compute_point_cloud

        self.initialized = True
        self.init_realsense(camera_serial)

        if not self.initialized:
            return

        self.pc = rs.pointcloud()

        self.trackers_pos = []

        if parameter:
            self.parameter = parameter
        else:
            try:
                with open(CONFIG_FILENAME, 'r') as fp:
                    json_parameters = json.load(fp)
                    self.parameter.update(json_parameters)
                    logger.info(f'Config file {CONFIG_FILENAME} found. Using parameters {self.parameter}')

            except FileNotFoundError:
                logger.warning('Config file {CONFIG_FILENAME} not found. Using default parameters {DEFAULT_CAMERA_PARAMS}')
                self.parameter.update(DEFAULT_CAMERA_PARAMS)
        
        default_param = self.parameter.copy()

        # Initialize the position estimation by reading the calibration file
        self.position_estimator = PositionEstimation(self.intr, configuration)
        self.position_estimator.intr= self.intr
        _, color_image, depth_image, _ = self.get_frame()
        self.position_estimator.compute_camera_to_simulation_transform()

        if not self.position_estimator.initialized:
            logger.error('Position estimation initialization failed. Using default parameters.')
            raise Exception('Position estimation initialization failed. Please check the camera calibration.')
        
        self.initialized = True

        if self.show_video_feed:        
            self.createFeedWindows()

        self.update() # to get a first frame and trackers


    def createFeedWindows(self):
        self.rootWindow = tk.Tk()
        self.rootWindow.resizable(False, False)

        self.rootWindow.title("Camera Feed Manager")
        ttk.Button(self.rootWindow, text="Close Windows", command=self.quit).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self.rootWindow, text="Save", command=lambda: json.dump(self.parameter, open(CONFIG_FILENAME, 'w'))).pack(side=tk.BOTTOM, padx=5, pady=5)	
        ttk.Button(self.rootWindow, text="Mask Window", command=self.createMaskWindow).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self.rootWindow, text="Frame Window", command=self.createFrameWindow).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self.rootWindow, text="HSV Window", command=self.createHSVWindow).pack(side=tk.BOTTOM, padx=5, pady=5)

        self.createMaskWindow()
        self.createFrameWindow()
        self.createHSVWindow()

        self.rootWindow.protocol("WM_DELETE_WINDOW", self.quit)
        self.rootWindow.update_idletasks()

    def createMaskWindow(self):
        if self.maskWindow is None or not self.maskWindow.running:
            self.maskWindow = CameraFeedWindow(rootWindow=self.rootWindow, trackbarParams=self.parameter, name='Mask')

    def createFrameWindow(self):
        if self.frameWindow is None or not self.frameWindow.running:
            self.frameWindow = CameraFeedWindow(rootWindow=self.rootWindow, name='RGB Frame')
    
    def createHSVWindow(self):
        if self.hsvWindow is None or not self.hsvWindow.running:
            self.hsvWindow = CameraFeedWindow(rootWindow=self.rootWindow, name='HSV')
    
    def quit(self):
        self.maskWindow.closed()
        self.frameWindow.closed()
        self.hsvWindow.closed()
        self.rootWindow.destroy()
        self.show_video_feed = False
        self.rootWindow = None

    def init_realsense(self, camera_serial: str=None):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if  camera_serial is not None:
            self.config.enable_device(camera_serial)

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        try:
            self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        except Exception as err:
            self.initialized = False
            raise Exception('DepthCamera', str(err))

        self.device = self.pipeline_profile.get_device()

        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        depth_sensor = self.device.first_depth_sensor()
        depth_sensor.set_option(rs.option.depth_units, 0.001)

        cfg = self.pipeline.start(self.config)

        self.profile = cfg.get_stream(rs.stream.depth)
        self.intr = self.profile.as_video_stream_profile().get_intrinsics()


    def calibrate(self):
        starttime = time.time()
        first = False
        success = False
        self.calibration_status = CalibrationStatusEnum.CALIBRATING

        # Create the windows to display the binrary mask and the HSV frame
        calibration_window = CameraFeedWindow(rootWindow=self.rootWindow, name='Calibration')

        while self.position_estimator.count_calibration_frames < 200 and time.time() - starttime < 300:
            self.position_estimator.intr= self.intr
            _, color_image, depth_image, _ = self.get_frame()
            success = self.position_estimator.calibrate(color_image, depth_image, first, calibration_window)
            first = success if not first else first
            self.rootWindow.update()

        if success:
            self.position_estimator.compute_camera_to_simulation_transform()
            logger.info(f"Camera {self.camera_serial} successfully calibrated.")

        # Close the calibration window
        calibration_window.closed()

        self.calibration_status = CalibrationStatusEnum.CALIBRATED if success else CalibrationStatusEnum.NOT_CALIBRATED
        return success


    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color

        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            return False, color_frame, depth_frame

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image, depth_image, depth_frame
    

    def update(self):
        ret, self.frame, self.depth_frame, depth_rsframe = self.get_frame()

        if ret is False:
            return
        # if frame is read correctly ret is True

        self.hsvFrame = cv.cvtColor(self.frame, cv.COLOR_BGR2HSV)

        # color definition
        red_lower = np.array([self.parameter['hue_l'], self.parameter['sat_l'], self.parameter['value_l']])
        red_upper = np.array([self.parameter['hue_h'], self.parameter['sat_h'], self.parameter['value_h']])

        # red color mask (sort of thresholding, actually segmentation)
        mask = cv.inRange(self.hsvFrame, red_lower, red_upper)
        mask2 = cv.inRange(self.depth_frame, 2, 430)

        mask = cv.bitwise_and(mask, mask2, mask=mask)

        erosion_shape = cv.MORPH_RECT
        erosion_size = self.parameter['erosion_size']
        element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                           (erosion_size, erosion_size))

        mask = cv.erode(mask, element, iterations=3)
        mask = cv.dilate(mask, element, iterations=3)

        self.maskFrame = cv.bitwise_and(self.frame, self.frame, mask=mask)

        if self.tracking:
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                areas = [cv.contourArea(cnt) for cnt in contours]

                self.trackers_pos = []
                for i, a in enumerate(areas):
                    if a > self.parameter['area']:
                        x, y = compute_contour_center(contours[i])
                        marker_mask = np.zeros_like(mask)

                        depth = compute_median_depth(contours[i], self.hsvFrame, self.depth_frame) if self.depth_frame[y, x] == 0 else self.depth_frame[y, x]
                        worldx, worldy, worldz = self.position_estimator.camera_image_to_simulation(x, y, depth)
                        self.trackers_pos.append([worldx, worldy, worldz])

                        cv.drawContours(marker_mask, [contours[i]], -1, color=255, thickness=-1)
                        for frame in [self.hsvFrame, self.frame]:
                            cv.circle(frame, (x, y), 2, color=255, thickness=-1)
                            cv.putText(frame, f"{i} ({x}, {y}, {depth})", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            cv.putText(frame, f"{i} ({worldx:.2f}, {worldy:.2f}, {worldz:.2f})", (x, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        
                        if self.show_video_feed:
                            cv.drawContours(self.frame, contours[i], -1, (255, 255, 0), 3)                

        if self.compute_point_cloud:
            points = self.pc.calculate(depth_rsframe)
            v = points.get_vertices()
            self.point_cloud = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

        if self.show_video_feed:
            if self.rootWindow is None:
                self.createFeedWindows()

            if self.maskWindow.running:
                self.maskWindow.set_frame(self.maskFrame)

            if self.frameWindow.running:
                self.frameWindow.set_frame(self.frame)

            if self.hsvWindow.running:
                self.hsvWindow.set_frame(self.hsvFrame)

            self.rootWindow.update()


    def close(self):
        if self.pipeline:
            self.pipeline.stop()
        if self.rootWindow:
            self.rootWindow.destroy()
        self.initialized = False


    def run_loop(self):
        while True:
            if self.rootWindow is None or not self.rootWindow.winfo_exists():
                break
            if self.show_video_feed:
                self.rootWindow.update()
            self.update()

        self.close()
