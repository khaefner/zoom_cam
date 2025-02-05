# zoom_cam
Uses GPU to do face recognition and create a virtual camera to keep your face zoomed in. AMD GPU.




Install the required libraries:

    OpenCV: pip install opencv-python
    PyVirtualCam: pip install pyvirtualcam

sudo apt install v4l2loopback-utils

sudo modprobe v4l2loopback video_nr=2 card_label="ZoomCam" exclusive_caps=1
