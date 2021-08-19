from enum import Enum
import numpy as np
from typing import Tuple

class HandIndices(Tuple[int, int], Enum):
    THUMB  = (0, 2  ) # 02 to 04
    INDEX  = (1, 6  ) # 06 to 08
    MIDDLE = (1, 10 ) # 10 to 12
    RING   = (1, 14 ) # 14 to 16
    PINKY  = (1, 18 ) # 18 to 20

class Directions(Enum):
    FRONT = 0
    BACK  = 1
    LEFT  = 2
    RIGHT = 3
    UP    = 4
    DOWN  = 5

class HandData:
    def __init__(self, threshold : float = 10.0, z_threshold : float = 0.001,
        move_threshold : float = 2.0, move_z_threshold : float = 0.00001):
        # raw positions -> can retrieve via hand indices enum
        self.__position = { 
            HandIndices.THUMB  : np.zeros((3, 1), dtype = float),
            HandIndices.INDEX  : np.zeros((3, 1), dtype = float),
            HandIndices.MIDDLE : np.zeros((3, 1), dtype = float),
            HandIndices.RING   : np.zeros((3, 1), dtype = float),
            HandIndices.PINKY  : np.zeros((3, 1), dtype = float) }
        #self.__offsets   = np.zeros((21, 2), dtype = float)
        self.__hand_pos = np.zeros((3), dtype = float)

        # reference data
        self.__prev_position = { 
            HandIndices.THUMB  : np.zeros((3, 1), dtype = float),
            HandIndices.INDEX  : np.zeros((3, 1), dtype = float),
            HandIndices.MIDDLE : np.zeros((3, 1), dtype = float),
            HandIndices.RING   : np.zeros((3, 1), dtype = float),
            HandIndices.PINKY  : np.zeros((3, 1), dtype = float) }
        self.__prev_hand_pos = np.zeros((3), dtype = float)
        #self.__reference_offsets = np.zeros((21, 2), dtype = float)

        # logic data
        self.__active = { 
            HandIndices.THUMB  : False,
            HandIndices.INDEX  : False,
            HandIndices.MIDDLE : False,
            HandIndices.RING   : False,
            HandIndices.PINKY  : False }
        self.__directions = {
            Directions.FRONT : False,
            Directions.BACK  : False,
            Directions.LEFT  : False,
            Directions.RIGHT : False,
            Directions.UP    : False,
            Directions.DOWN  : False,
        }

        self.__threshold = threshold
        self.__z_threshold = z_threshold
        self.__move_threshold = move_threshold
        self.__move_z_threshold = move_z_threshold

    @property
    def hand_pos(self):
        return self.__hand_pos

    @hand_pos.setter
    def hand_pos(self, hand_pos):
        self.__hand_pos = hand_pos

    @property
    def prev_hand_pos(self):
        return self.__prev_hand_pos

    @prev_hand_pos.setter
    def prev_hand_pos(self, prev_hand_pos):
        self.__prev_hand_pos = prev_hand_pos

    @property
    def active(self):
        return self.__active

    @active.setter
    def active(self, active):
        self.__active = active

    @property
    def directions(self):
        return self.__directions

    @directions.setter
    def directions(self, directions):
        self.__directions = directions

    def position(self, x_pos, y_pos):
        temparray = np.zeros((21, 2), dtype = float)
        index = 0
        for x, y in zip(x_pos, y_pos):
            temparray[index] = [x, y]
            index += 1
        temparray = temparray.astype("float64")

        for hand_idx in HandIndices:
            coord_idx = hand_idx.value[0]
            num_idx = hand_idx.value[1]
            for add in range(3):
                if abs(temparray[num_idx + add][coord_idx] - self.__position[hand_idx][add]) > self.__threshold:
                    self.__prev_position[hand_idx][add], self.__position[hand_idx][add] = self.__position[hand_idx][add], temparray[num_idx + add][coord_idx]

    def hand_position(self, x_pos, y_pos, z_pos):
        temparray = np.zeros((3), dtype = float)
        temparray[0] = sum(x_pos) / len(x_pos)
        temparray[1] = sum(y_pos) / len(y_pos)
        temparray[2] = sum(z_pos) / len(z_pos)

        for i in range(2):
            if abs(temparray[i] - self.__hand_pos[i]) > self.__threshold:
                self.__prev_hand_pos[i], self.__hand_pos[i] = self.__hand_pos[i], temparray[i]
        if abs(temparray[2] - self.__hand_pos[2]) > self.__z_threshold:
                self.__prev_hand_pos[2], self.__hand_pos[2] = self.__hand_pos[2], temparray[2]

    def activity(self):
        for hand_index, position_set in self.__position.items():
            self.__active[hand_index] = (position_set[1] < position_set[0]) and (position_set[2] < position_set[0])

    def movement(self):
        x_axis = self.__hand_pos[0] - self.__prev_hand_pos[0]
        self.__directions[Directions.LEFT]  = x_axis < -self.__move_threshold
        self.__directions[Directions.RIGHT] = x_axis >  self.__move_threshold

        y_axis = self.__hand_pos[1] - self.__prev_hand_pos[1]
        self.__directions[Directions.UP]   = y_axis < -self.__move_threshold
        self.__directions[Directions.DOWN] = y_axis >  self.__move_threshold

        z_axis = self.__hand_pos[2] - self.__prev_hand_pos[2]
        self.__directions[Directions.FRONT] = z_axis < -self.__move_z_threshold
        self.__directions[Directions.BACK]  = z_axis > self.__move_z_threshold

    # catered specifically for HandDetect hand_details
    def update(self, all_positions_data):
        x_pos = [sublist[2] for sublist in all_positions_data]
        y_pos = [sublist[3] for sublist in all_positions_data]
        z_pos = [sublist[4] for sublist in all_positions_data]
        self.position(x_pos, y_pos)
        self.hand_position(x_pos, y_pos, z_pos)
        self.activity()
        self.movement()

    def __repr__(self):
        hand_activity_str = ("Hand Activity:\n\t" +
            "Thumb: "  + str(self.__active[HandIndices.THUMB])  + "\n\t" +
            "Index: "  + str(self.__active[HandIndices.INDEX])  + "\n\t" +
            "Middle: " + str(self.__active[HandIndices.MIDDLE]) + "\n\t" +
            "Ring: "   + str(self.__active[HandIndices.RING])   + "\n\t" +
            "Pinky: "  + str(self.__active[HandIndices.PINKY]))
        hand_movement_str = ("Hand Movement:\n\t" +
            "Front: "  + str(self.__directions[Directions.FRONT]) + "\n\t" +
            "Back: "  + str(self.__directions[Directions.BACK])  + "\n\t" +
            "Left: " + str(self.__directions[Directions.LEFT])  + "\n\t" +
            "Right: "   + str(self.__directions[Directions.RIGHT]) + "\n\t" +
            "Up: "   + str(self.__directions[Directions.UP])    + "\n\t" +
            "Down: "  + str(self.__directions[Directions.DOWN ]))
        prev_hand_pos_str = "Previous Hand Position: (%5.2f, %5.2f, %5.2f)" % (self.__prev_hand_pos[0], self.__prev_hand_pos[1], self.__prev_hand_pos[2])
        hand_pos_str = "Hand Position: (%5.2f, %5.2f, %5.2f)" % (self.__hand_pos[0], self.__hand_pos[1], self.__hand_pos[2])
        previous_pos_str = "Previous Positions:\n\t" + "\n\t".join(
            ("%-18s + %d: (%5.2f)" % (
            str(hand_index), index, prev_xy))
            for hand_index in HandIndices
            for index, prev_xy in enumerate(self.__prev_position[hand_index]))
        pos_str = "Previous Positions:\n\t" + "\n\t".join(
            ("%-18s + %d: (%5.2f)" % (
            str(hand_index), index, curr_xy))
            for hand_index in HandIndices
            for index, curr_xy in enumerate(self.__prev_position[hand_index]))

        return (hand_activity_str + "\n\n" + 
                hand_movement_str + "\n\n" +
                prev_hand_pos_str + "\n\n" +
                hand_pos_str      + "\n\n" +
                previous_pos_str  + "\n\n" + 
                pos_str + "\n\n")

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
        elif key == ord("w"): print(sum(hand_data.active.values()))
