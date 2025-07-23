import os
import json
import logging
from time import sleep
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2 as cv
import pyrealsense2 as rs

from ._camerafeedwindow import CameraFeedWindow

FORMAT = "[%(levelname)s]\t[%(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILENAME = os.path.dirname(__file__) + '/cameraparameter.json'


def convert_depth_pixel_to_metric_coordinate(depth, pixel_x, pixel_y, camera_intrinsics):
    """
    Convert the depth and image point information to metric coordinates

    Parameters:
    -----------
    depth 	 	 	 : double
                                               The depth value of the image point
    pixel_x 	  	 	 : double
                                               The x value of the image coordinate
    pixel_y 	  	 	 : double
                                                    The y value of the image coordinate
    camera_intrinsics : The intrinsic values of the imager in whose coordinate
                        system the depth_frame is computed

        Return:
    ----------
    X : double
            The x value in meters
    Y : double
            The y value in meters
    Z : double
            The z value in meters

    """

    X = (pixel_x - camera_intrinsics.ppx) / camera_intrinsics.fx * depth
    Y = (pixel_y - camera_intrinsics.ppy) / camera_intrinsics.fy * depth
    return [X, Y, depth]


def compute_cdg(contour):
    M = cv.moments(contour)
    cX = 0
    cY = 0
    if M['m00'] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    return cX, cY


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
    parameter = {}
    tracking = True
    trackers_pos = []
    maskWindow = None
    hsvWindow = None
    rootWindow = None
    hsvFrame = None
    maskFrame = None

    @property
    def camera_serial(self) -> str:
        """
        Returns the serial of the camera as str
        
        """
        return self.device.get_info(rs.camera_info.serial_number)


    def __init__(self, camera_serial: str=None, parameter: dict=None, compute_point_cloud: bool=False, show_video_feed: bool=False, tracking: bool=True) -> None:
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
                logger.debug(f"Opening config file {CONFIG_FILENAME}")
                with open(CONFIG_FILENAME, 'r') as fp:
                    json_parameters = json.load(fp)
                    self.parameter.update(json_parameters)
                    logger.info(f'Config file {CONFIG_FILENAME} found. Using parameters {self.parameter}')

            except FileNotFoundError:
                logger.warning('Config file {CONFIG_FILENAME} not found. Using default parameters {"hue_h": 90, "hue_l": 36, "sat_h": 255, "sat_l": 138, "value_h": 255, "value_l": 35, "erosion_size": 0, "area": 100}')
                self.parameter.update({"hue_h": 90, "hue_l": 36, "sat_h": 255, "sat_l": 138, "value_h": 255, "value_l": 35, "erosion_size": 0, "area": 100})
        
        default_param = self.parameter.copy()


        logger.debug(f'Camera show_video_feed: {self.show_video_feed}')

        if self.show_video_feed:        
            self.createWindows()

        self.update() # to get a first frame


    def createWindows(self):
        self.rootWindow = tk.Tk()
        self.rootWindow.resizable(False, False)
        # self.rootWindow.tk.call("source", os.path.abspath("../../parts\controllers/azure_ttk_theme/azure.tcl")) # https://github.com/rdbende/Azure-ttk-theme
        # self.rootWindow.tk.call("set_theme", "light")

        self.rootWindow.title("Camera Feed Manager")
        ttk.Button(self.rootWindow, text="Close Windows", command=self.quit).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self.rootWindow, text="Save", command=lambda: json.dump(self.parameter, open(CONFIG_FILENAME, 'w'))).pack(side=tk.BOTTOM, padx=5, pady=5)	
        ttk.Button(self.rootWindow, text="Mask Window", command=self.createMaskWindow).pack(side=tk.BOTTOM, padx=5, pady=5)
        ttk.Button(self.rootWindow, text="HSV Window", command=self.createHSVWindow).pack(side=tk.BOTTOM, padx=5, pady=5)

        self.createMaskWindow()
        self.createHSVWindow()

        self.rootWindow.protocol("WM_DELETE_WINDOW", self.quit)
        self.rootWindow.update_idletasks()

    def createMaskWindow(self):
        if self.maskWindow is None or not self.maskWindow.running:
            self.maskWindow = CameraFeedWindow(rootWindow=self.rootWindow, trackbarParams=self.parameter, name='Mask')

    def createHSVWindow(self):
        if self.hsvWindow is None or not self.hsvWindow.running:
            self.hsvWindow = CameraFeedWindow(rootWindow=self.rootWindow, name='HSV')
    
    def quit(self):
        self.maskWindow.closed()
        self.hsvWindow.closed()
        self.rootWindow.destroy()
        self.show_video_feed = False
        self.rootWindow = None

    def init_realsense(self, camera_serial=None):
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

    def get_frame(self):
        # Wait for a coherent pair of frames: depth and color

        frames = self.pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            logger.debug('no frame')
            return False, color_frame, depth_frame

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return True, color_image, depth_image, depth_frame

    def update(self):
        ret, frame, depth_image, depth_frame = self.get_frame()

        if ret is False:
            return
        # if frame is read correctly ret is True

        self.hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # color definition
        red_lower = np.array(
            [self.parameter['hue_l'], self.parameter['sat_l'], self.parameter['value_l']])
        red_upper = np.array(
            [self.parameter['hue_h'], self.parameter['sat_h'], self.parameter['value_h']])

        # red color mask (sort of thresholding, actually segmentation)
        mask = cv.inRange(self.hsvFrame, red_lower, red_upper)
        mask2 = cv.inRange(depth_image, 2, 430)

        mask = cv.bitwise_and(mask, mask2, mask=mask)

        erosion_shape = cv.MORPH_RECT
        erosion_size = self.parameter['erosion_size']
        element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                           (erosion_size, erosion_size))

        mask = cv.erode(mask, element, iterations=3)
        mask = cv.dilate(mask, element, iterations=3)

        self.maskFrame = cv.bitwise_and(frame, frame, mask=mask)

        if self.tracking:
            contours, _ = cv.findContours(
                mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0:
                areas = [cv.contourArea(cnt) for cnt in contours]

                self.trackers_pos = []
                for i, a in enumerate(areas):
                    if a > self.parameter['area']:
                        x, y = compute_cdg(contours[i])
                        self.trackers_pos.append(convert_depth_pixel_to_metric_coordinate(
                            (depth_image[y, x]), x, y, self.intr))
                        
                        if self.show_video_feed:
                            cv.drawContours(self.hsvFrame, contours[i], -1, (255, 255, 0), 3)                    

        if self.compute_point_cloud:
            points = self.pc.calculate(depth_frame)
            v = points.get_vertices()
            self.point_cloud = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

        if self.show_video_feed:
            if self.rootWindow is None:
                self.createWindows()

            if self.maskWindow.running:
                self.maskWindow.set_frame(self.maskFrame)
            # image = ImageTk.PhotoImage(image=Image.fromarray(cv.cvtColor(hsv, cv.COLOR_BGR2RGB)))

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
