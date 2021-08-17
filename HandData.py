from enum import Enum
import numpy as np

class HandIndices(Enum):
    HAND   = 0 # 00 to 20
    THUMB  = 1 # 02 to 04
    INDEX  = 2 # 05 to 08
    MIDDLE = 3 # 09 to 12
    RING   = 4 # 13 to 16
    PINKY  = 5 # 17 to 20

class HandData:
    def __init__(self, threshold : float = 10.0):
        # raw positions -> can retrieve via hand indices enum
        self._position  = np.zeros((6, 2), dtype = float)
        self._offsets   = np.zeros((6, 2), dtype = float)

        # reference data
        self._prev_position = np.zeros((6, 2), dtype = float)
        self._init_offsets = np.zeros((6, 2), dtype = float)

        self._threshold = threshold

    def position(self, x_pos, y_pos):
        temparray = np.array([
            [ sum(x_pos) / len(x_pos), sum(y_pos) / len(y_pos) ], 
            [ x_pos[4], y_pos[4] ], [ x_pos[8], y_pos[8] ], 
            [ x_pos[12], y_pos[12] ], [ x_pos[16], y_pos[16] ], 
            [ x_pos[20], y_pos[20] ]])
        temparray = temparray.astype("float64")

        for i, xy in enumerate(self._position):
            if abs(temparray[i][0] - xy[0]) > self._threshold:
                self._prev_position[i][0] = xy[0]
                self._position[i][0] = temparray[i][0]
            if abs(temparray[i][1] - xy[1]) > self._threshold:
                self._prev_position[i][1] = xy[1]
                self._position[i][1] = temparray[i][1]

    def offset(self, x_offset, y_offset):
        self._offsets = np.array([
            [ sum(x_offset) / len(x_offset), sum(y_offset) / len(y_offset) ], 
            [ x_offset[4], y_offset[4] ], [ x_offset[8], y_offset[8] ], 
            [ x_offset[12], y_offset[12] ], [ x_offset[16], y_offset[16] ], 
            [ x_offset[20], y_offset[20] ]])
        self._offsets = self._offsets.astype("float64")

    # catered specifically for HandDetect hand_details
    def update(self, all_positions_data):
        x_pos = [sublist[2] for sublist in all_positions_data]
        y_pos = [sublist[3] for sublist in all_positions_data]
        self.position(x_pos, y_pos)
        self.offset(
            [x - x_pos[0] if i > 0 else 0.0 for i, x in enumerate(x_pos)], 
            [y - y_pos[0] if i > 0 else 0.0 for i, y in enumerate(y_pos)])

    def recalibrate_offset(self, all_positions_data):
        x_pos = [sublist[2] for sublist in all_positions_data]
        y_pos = [sublist[3] for sublist in all_positions_data]

        x_offset = [x - x_pos[0] if i > 0 else 0.0 for i, x in enumerate(x_pos)]
        y_offset = [y - y_pos[0] if i > 0 else 0.0 for i, y in enumerate(y_pos)]

        self._init_offsets = np.array([
            [ sum(x_offset) / len(x_offset), sum(y_offset) / len(y_offset) ], 
            [ x_offset[4], y_offset[4] ], [ x_offset[8], y_offset[8] ], 
            [ x_offset[12], y_offset[12] ], [ x_offset[16], y_offset[16] ], 
            [ x_offset[20], y_offset[20] ]])
        self._init_offsets = self._init_offsets.astype("float64")

    def __repr__(self):
        return ("Initial Offsets:\n\t" + 
            "Hand: "   + str(self._init_offsets[HandIndices.HAND.value]) +
            "Thumb: "  + str(self._init_offsets[HandIndices.THUMB.value]) +
            "Index: "  + str(self._init_offsets[HandIndices.INDEX.value]) +
            "Middle: " + str(self._init_offsets[HandIndices.MIDDLE.value]) +
            "Ring: "   + str(self._init_offsets[HandIndices.RING.value]) +
            "Pinky: "  + str(self._init_offsets[HandIndices.PINKY.value])
            + "\n" + "Current Offsets:\n\t" + 
            "Hand: "   + str(self._offsets[HandIndices.HAND.value]) +
            "Thumb: "  + str(self._offsets[HandIndices.THUMB.value]) +
            "Index: "  + str(self._offsets[HandIndices.INDEX.value]) +
            "Middle: " + str(self._offsets[HandIndices.MIDDLE.value]) +
            "Ring: "   + str(self._offsets[HandIndices.RING.value]) +
            "Pinky: "  + str(self._offsets[HandIndices.PINKY.value])

            + "\n" + "Previous Positions:\n\t" + 
            "Hand: "   + str(self._prev_position[HandIndices.HAND.value]) +
            "Thumb: "  + str(self._prev_position[HandIndices.THUMB.value]) +
            "Index: "  + str(self._prev_position[HandIndices.INDEX.value]) +
            "Middle: " + str(self._prev_position[HandIndices.MIDDLE.value]) +
            "Ring: "   + str(self._prev_position[HandIndices.RING.value]) +
            "Pinky: "  + str(self._prev_position[HandIndices.PINKY.value])
            + "\n" + "Current Positions:\n\t" + 
            "Hand: "   + str(self._position[HandIndices.HAND.value]) +
            "Thumb: "  + str(self._position[HandIndices.THUMB.value]) +
            "Index: "  + str(self._position[HandIndices.INDEX.value]) +
            "Middle: " + str(self._position[HandIndices.MIDDLE.value]) +
            "Ring: "   + str(self._position[HandIndices.RING.value]) +
            "Pinky: "  + str(self._position[HandIndices.PINKY.value]))

if __name__ == "__main__":
    import Camera as cam
    import HandDetect as hd
    import Renderer as rdr
    import cv2
  
    camera_object = cam.Camera(camera_index = 1)
    hand_detector = hd.HandDetect(max_num_hands = 1)
    renderer = rdr.Renderer()
    hand_data = HandData()
  
    while True:
        if not camera_object.capture(): break
  
        main_image_rgb = camera_object.convert_image()
        success = hand_detector(main_image_rgb, camera_object.image_height, camera_object.image_width)
        if not success: continue
  
        hand_data.update(hand_detector.get_details)

        hand_detector.render(camera_object.frame(), renderer.render_mp, renderer.render_cv2)
  
        camera_object.display_image()
  
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
        elif key == ord("q"): print(repr(hand_data))
        elif key == ord("w"): hand_data.recalibrate_offset(hand_detector.get_details)
