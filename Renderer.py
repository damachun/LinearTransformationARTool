import mediapipe as mp
import cv2
from typing import Tuple

class Renderer:
    def __init__(self, 
      # mediapipe
    mp_line_color : Tuple[int, int, int] = (255, 0, 0),
    mp_line_thickness : int = 1,
    mp_circle_color : Tuple[int, int, int] = (0, 0, 255),
    mp_circle_thickness : int = 1,
    mp_circle_radius : int = 1,
      # opencv
    cv2_text_font = cv2.FONT_HERSHEY_PLAIN,
    cv2_text_scale : float = 1.0,
    cv2_text_color : Tuple[int, int, int] = (0, 0, 255),
    cv2_text_thickness : int = 1):

        # mediapipe renderer
        self.mp_renderer     = mp.solutions.drawing_utils
          # mediapipe line
        self.mp_line_specs   = self.mp_renderer.DrawingSpec(mp_line_color,  mp_line_thickness)
          # mediapipe circle
        self.mp_circle_specs = self.mp_renderer.DrawingSpec(mp_circle_color,    mp_circle_thickness, mp_circle_radius)

        # opencv renderer
          # opencv text
        self.cv2_text_font      = cv2_text_font
        self.cv2_text_scale     = cv2_text_scale
        self.cv2_text_color     = cv2_text_color
        self.cv2_text_thickness = cv2_text_thickness

  
    def __repr__(self):
        return "Why"

    def edit_mp_line(self,
    mp_line_color : Tuple[int, int, int] = (255, 0, 0),
    mp_line_thickness : int = 1):
        self.mp_line_specs = self.mp_renderer.DrawingSpec(mp_line_color, mp_line_thickness)

    def edit_mp_circle(self, 
    mp_circle_color : Tuple[int, int, int] = (0, 0, 255),
    mp_circle_thickness : int = 1,
    mp_circle_radius : int = 1):
        self.mp_circle_specs = self.mp_renderer.DrawingSpec(mp_circle_color, mp_circle_thickness, mp_circle_radius)

    def edit_cv2_text(self, cv2_text_font = cv2.FONT_HERSHEY_PLAIN,
    cv2_text_scale : float = 1.0,
    cv2_text_color : Tuple[int, int, int] = (0, 0, 255),
    cv2_text_thickness : int = 1):
        self.cv2_text_font      = cv2_text_font
        self.cv2_text_scale     = cv2_text_scale
        self.cv2_text_color     = cv2_text_color
        self.cv2_text_thickness = cv2_text_thickness

    def render_mp(self, main_image, landmarks, flags):
        self.mp_renderer.draw_landmarks(main_image, landmarks, flags,
        self.mp_line_specs, self.mp_circle_specs)

    def render_cv2(self, main_image, text, position):
        cv2.putText(main_image, text, position, self.cv2_text_font,
        self.cv2_text_scale, self.cv2_text_color, self.cv2_text_thickness)

if __name__ == "__main__":
    import HandDetect as hd
  
    capture_object = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    hand_detector = hd.HandDetect()
    renderer = Renderer()
  
    while True:
        success, main_image = capture_object.read()
        if not success: break
  
        main_image_rgb = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
        image_height = main_image.shape[0]
        image_width = main_image.shape[1]
        success = hand_detector(main_image_rgb, image_height, image_width)
        if not success: continue
  
        hand_detector.render(main_image, renderer.render_mp, renderer.render_cv2)
  
        cv2.imshow("Hand Detector Output", main_image)
  
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
  
    cv2.destroyAllWindows()