
import cv2
import numpy as np

class Camera:
    def __init__(self, camera_index : int = 0, window_name : str = "Frame"):

        # capture details
        self._capture_object = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self._window_name = window_name
        cv2.namedWindow(window_name)
        self._frame = np.zeros(())

        self._image_height = 0
        self._image_width = 0

    def __del__(self):
        if self._window_name:
            cv2.destroyWindow(self._window_name)

    def capture(self) -> bool:

        retval, self._frame = self._capture_object.read()
        if retval:
            self._image_height, self._image_width, _ = self._frame.shape
        return retval

    def convert_image(self, code : int = cv2.COLOR_BGR2RGB):

        return cv2.cvtColor(self._frame, code)

    # non const reference, modifiable
    def frame(self):
        return self._frame

    @property
    def image_height(self):
        return self._image_height

    @image_height.setter
    def image_height(self, image_height):
        self._image_height = image_height

    @property
    def image_width(self):
        return self._image_width

    @image_width.setter
    def image_width(self, image_width):
        self._image_width = image_width

    def display_image(self):
        cv2.imshow(self._window_name, self._frame)

if __name__ == "__main__":
    import HandDetect as hd
    import Renderer as rdr
  
    camera_object = Camera(camera_index = 1)
    hand_detector = hd.HandDetect()
    renderer = rdr.Renderer()
  
    while True:
        if not camera_object.capture(): break
  
        main_image_rgb = camera_object.convert_image()
        success = hand_detector(main_image_rgb, camera_object.image_height, camera_object.image_width)
        if not success: continue
  
        hand_detector.render(camera_object.frame(), renderer.render_mp, renderer.render_cv2)
  
        camera_object.display_image()
  
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break