import mediapipe as mp

# functor to detect hands
class handDetect():
    def __init__(self, static_image_mode : bool = False, 
        max_num_hands : int = 2,
        min_detection_confidence : float = 0.5, 
        min_tracking_confidence : float = 0.5):

        # required for mp.solutions.hands.Hands()
        # see if it actually needs to be saved
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # mediapipe hand objects
        self.hand_pipeline = mp.solutions.hands
        self.hand_detector = self.hand_pipeline.Hands(
            static_image_mode, max_num_hands,
            min_detection_confidence, min_tracking_confidence)

        # output objects
        #self.hand_processed
        self.hand_details = []

    def __call__(self, main_image_rgb, height : int, width : int) -> bool:

        # process image to get hand
        self.hand_processed = self.hand_detector.process(main_image_rgb)

        # edge check
        if not self.hand_processed.multi_hand_landmarks:
            return False

        # update hand details list
        self.hand_details.clear()
        for hand_index, hand_landmarks in enumerate(self.hand_processed.multi_hand_landmarks):
            for landmark_id, landmarks in enumerate(hand_landmarks.landmark):
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                self.hand_details.append([hand_index, landmark_id, x, y])
        
        return True

    def render(self, main_image, render_fn_hand, render_fn_details):

        # edge check
        if not self.hand_processed.multi_hand_landmarks:
            return
        
        # render hand landmark indicator
        for landmarks in self.hand_processed.multi_hand_landmarks:
            render_fn_hand(main_image, landmarks, 
                self.hand_pipeline.HAND_CONNECTIONS)

        # render details above
        for _, landmark_id, x, y in self.hand_details:
            render_fn_details(main_image, str(int(landmark_id)), (x + 10, y + 10))

    # returns 2d array, each element contains (hand index, id, coordinates)
    # const it
    @property
    def get_details(self):
        return self.hand_details

if __name__ == "__main__":
    print("Hello")
    import cv2

    def render_fn_hand(main_image, landmarks, flags):
        mp.solutions.drawing_utils.draw_landmarks(main_image, landmarks, flags)

    def render_fn_details(main_image, text, position):
        cv2.putText(main_image, text, position, cv2.FONT_HERSHEY_PLAIN,
            1, (0, 255, 0), 2)

    capture_object = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    hand_detector = handDetect()

    while True:
        success, main_image = capture_object.read()
        if not success: break

        main_image_rgb = cv2.cvtColor(main_image, cv2.COLOR_BGR2RGB)
        image_height = main_image.shape[0]
        image_width = main_image.shape[1]
        success = hand_detector(main_image_rgb, image_height, image_width)
        if not success: continue

        hand_detector.render(main_image, render_fn_hand, render_fn_details)

        cv2.imshow("Hand Detector Output", main_image)

        if cv2.waitKey(2) & 0xFF == 27: break

    
