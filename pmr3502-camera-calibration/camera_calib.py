from picamera2 import Picamera2
from matplotlib import pyplot as plt
from IPython.display import clear_output
import os

os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
Picamera2.set_logging(Picamera2.ERROR)

with Picamera2(tuning=os.environ.get('LIBCAMERA_RPI_TUNING_FILE', None)) as camera:
    try:
        cfg = camera.create_video_configuration(main={"size": (1296, 972), "format": "BGR888"})
        camera.configure(cfg)
        camera.start()
        while True:
            img = camera.capture_array()
            clear_output(wait=True)
            plt.imshow(img)
            plt.show()
    except KeyboardInterrupt:
        pass