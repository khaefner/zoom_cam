sudo modprobe v4l2loopback video_nr=2 card_label="VirtualCam" exclusive_caps=1
v4l2-ctl -d /dev/video2 --set-fmt-video=width=1280,height=720
