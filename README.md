# Emio API

Emio API is a simple and easy-to-use API for controling the Emio robot from [Compliance Robotics](https://compliance-robotics.com/).

The `emioapi` package is published on [PyPI](https://pypi.org/project/emioapi/).

## Installation
To install the Emio API, you can use pip:

```bash
python -m pip install emioapi
```

or to get the latest version that is unstable

```bash
python -m pip install git+https://github.com/SofaComplianceRobotics/Emio.API.git@release-main
```

## Usage
The Emio API provides the `EmioAPI` class, which can be used to control the Emio robot. The API provides methods for controlling the robot's motors and camera.
You can look at the [motors_example.py](motors_example.py) file for a simple example of how to use the API to control the motors of the Emio robot.
To control the camera, look at the [camera_example.py](exampes/camera_example.py) file or [multiprocess_camera_example.py](examples/multiprocess_camera_example.py).


Simple example thaht sets the angles of the motors to 0 radians, waits for 1 second, and then prints the status of the robot:
```python
from emioapi import emio

# Open a port to the Emio robot and configure it
emio.connectToEmioDevice()

emio.motors.angles = [0] * 4  # Set the angles of the motors to 0 radians
time.sleep(1)  # Wait for 1 second
emio.printStatus() # Print the status of the robot
emio.disconnect()  # Close the connection to the robot
```

## For Developers
The documentation is generated using [pydoc-markdown](https://pypi.org/project/pydoc-markdown/). To generate the documentation, you need to install `pydoc-markdown`:

```bash
pipx install pydoc-markdown
```

Then, you can generate the documentation in a `emioapi.md` file, by using the following command at the root of the project:

```bash
pydoc-markdown
```
