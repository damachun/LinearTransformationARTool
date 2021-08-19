from enum import Enum
import numpy as np

class HandIndices(Enum):
    HAND   = 0 # 00 to 20
    THUMB  = 4 # 02 to 04
    INDEX  = 8 # 05 to 08
    MIDDLE = 12 # 09 to 12
    RING   = 16 # 13 to 16
    PINKY  = 20 # 17 to 20

class HandData:
    def __init__(self, threshold : float = 10.0):
        # raw positions -> can retrieve via hand indices enum
        self.__position  = np.zeros((21, 2), dtype = float)
        #self.__offsets   = np.zeros((21, 2), dtype = float)

        # reference data
        self.__prev_position = np.zeros((21, 2), dtype = float)
        #self.__reference_offsets = np.zeros((21, 2), dtype = float)

        self.__threshold = threshold

    def position(self, x_pos, y_pos):
        temparray = np.zeros((21, 2), dtype = float)
        index = 1
        temparray[0] = [ sum(x_pos) / len(x_pos), sum(y_pos) / len(y_pos)]
        for x, y in zip(x_pos, y_pos):
            temparray[index] = [x, y]
            index += 1
        temparray = temparray.astype("float64")

        for i, xy in enumerate(self.__position):
            if abs(temparray[i][0] - xy[0]) > self.__threshold:
                self.__prev_position[i][0] = xy[0]
                self.__position[i][0] = temparray[i][0]
            if abs(temparray[i][1] - xy[1]) > self.__threshold:
                self.__prev_position[i][1] = xy[1]
                self.__position[i][1] = temparray[i][1]

    #def offset(self, x_pos, y_pos):
    #    self.__offsets = np.array([
    #        [ 0.0, 0.0 ], 
    #        [ x_pos[4] - x_pos[2], y_pos[4] - y_pos[2] ], 
    #        [ x_pos[8] - x_pos[5], y_pos[8] - y_pos[5] ], 
    #        [ x_pos[12] - x_pos[9], y_pos[12] - y_pos[9] ], 
    #        [ x_pos[16] - x_pos[13], y_pos[16] - y_pos[13] ], 
    #        [ x_pos[17] - x_pos[17], y_pos[20] - y_pos[17] ]])
    #    self.__offsets = self.__offsets.astype("float64")

    # catered specifically for HandDetect hand_details
    def update(self, all_positions_data):
        x_pos = [sublist[2] for sublist in all_positions_data]
        y_pos = [sublist[3] for sublist in all_positions_data]
        self.position(x_pos, y_pos)

        #x_offset = [x - x_pos[0] if i > 0 else 0.0 for i, x in enumerate(x_pos)]
        #y_offset = [y - y_pos[0] if i > 0 else 0.0 for i, y in enumerate(y_pos)]
        #self.offset(x_pos, y_pos)

    #def recalibrate_offset(self, all_positions_data):
    #    x_pos = [sublist[2] for sublist in all_positions_data]
    #    y_pos = [sublist[3] for sublist in all_positions_data]
    #
    #    self.__reference_offsets = np.array([
    #        [ 0.0, 0.0 ], 
    #        [ x_pos[4] - x_pos[2], y_pos[4] - y_pos[2] ], 
    #        [ x_pos[8] - x_pos[5], y_pos[8] - y_pos[5] ], 
    #        [ x_pos[12] - x_pos[9], y_pos[12] - y_pos[9] ], 
    #        [ x_pos[16] - x_pos[13], y_pos[16] - y_pos[13] ], 
    #        [ x_pos[17] - x_pos[17], y_pos[20] - y_pos[17] ]])
    #    self.__reference_offsets = self.__reference_offsets.astype("float64")

    def finger_active(self, finger_index = HandIndices.HAND.value):
        return (np.sign(self.__offsets[finger_index][0]) == 
            np.sign(self.__reference_offsets[finger_index][0])) and (np.sign(
            self.__offsets[finger_index][1]) == 
            np.sign(self.__reference_offsets[finger_index][1]))

    def num_of_fingers_active(self) -> int:
        count = 0

        for i in range(1,6):
            if self.finger_active(i): count += 1

        return count

    def __repr__(self):
        return (
            #"Reference Offsets:\n\t" + 
            #"Hand: "   + "({:5.2f}, {:5.2f})".format(
            #    self.__reference_offsets[0][0], self.__reference_offsets[0][1]) #+ "\n\t" +
            #"Thumb: "  + "({:5.2f}, {:5.2f})".format(
            #    self.__reference_offsets[1][0], self.__reference_offsets[1][1]) #+ "\n\t" +
            #"Index: "  + "({:5.2f}, {:5.2f})".format(
            #    self.__reference_offsets[2][0], self.__reference_offsets[2][1]) #+ "\n\t" +
            #"Middle: " + "({:5.2f}, {:5.2f})".format(
            #    self.__reference_offsets[3][0], self.__reference_offsets[3][1]) #+ "\n\t" +
            #"Ring: "   + "({:5.2f}, {:5.2f})".format(
            #    self.__reference_offsets[4][0], self.__reference_offsets[4][1]) #+ "\n\t" +
            #"Pinky: "  + "({:5.2f}, {:5.2f})".format(
            #    self.__reference_offsets[5][0], self.__reference_offsets[5][1]) #+ "\n\n" +
            #
            #"Current Offsets:\n\t" + 
            #"Hand: "   + "({:5.2f}, {:5.2f})".format(
            #    self.__offsets[0][0], self.__offsets[0][1]) + "\n\t" +
            #"Thumb: "  + "({:5.2f}, {:5.2f})".format(
            #    self.__offsets[1][0], self.__offsets[1][1]) + "\n\t" +
            #"Index: "  + "({:5.2f}, {:5.2f})".format(
            #    self.__offsets[2][0], self.__offsets[2][1]) + "\n\t" +
            #"Middle: " + "({:5.2f}, {:5.2f})".format(
            #    self.__offsets[3][0], self.__offsets[3][1]) + "\n\t" +
            #"Ring: "   + "({:5.2f}, {:5.2f})".format(
            #    self.__offsets[4][0], self.__offsets[4][1]) + "\n\t" +
            #"Pinky: "  + "({:5.2f}, {:5.2f})".format(
            #    self.__offsets[5][0], self.__offsets[5][1]) + "\n\n" +

            "Previous Positions:\n\t" + 
            "Hand: "   + "({:5.2f}, {:5.2f})".format(
                self.__prev_position[0][0], self.__prev_position[0][1]) + "\n\t" +
            "Thumb: "  + "({:5.2f}, {:5.2f})".format(
                self.__prev_position[1][0], self.__prev_position[1][1]) + "\n\t" +
            "Index: "  + "({:5.2f}, {:5.2f})".format(
                self.__prev_position[2][0], self.__prev_position[2][1]) + "\n\t" +
            "Middle: " + "({:5.2f}, {:5.2f})".format(
                self.__prev_position[3][0], self.__prev_position[3][1]) + "\n\t" +
            "Ring: "   + "({:5.2f}, {:5.2f})".format(
                self.__prev_position[4][0], self.__prev_position[4][1]) + "\n\t" +
            "Pinky: "  + "({:5.2f}, {:5.2f})".format(
                self.__prev_position[5][0], self.__prev_position[5][1]) + "\n\n" +

            "Current Positions:\n\t" + 
            "Hand: "   + "({:5.2f}, {:5.2f})".format(
                self.__position[0][0], self.__position[0][1]) + "\n\t" +
            "Thumb: "  + "({:5.2f}, {:5.2f})".format(
                self.__position[1][0], self.__position[1][1]) + "\n\t" +
            "Index: "  + "({:5.2f}, {:5.2f})".format(
                self.__position[2][0], self.__position[2][1]) + "\n\t" +
            "Middle: " + "({:5.2f}, {:5.2f})".format(
                self.__position[3][0], self.__position[3][1]) + "\n\t" +
            "Ring: "   + "({:5.2f}, {:5.2f})".format(
                self.__position[4][0], self.__position[4][1]) + "\n\n" +
            "Pinky: "  + "({:5.2f}, {:5.2f})".format(
                self.__position[5][0], self.__position[5][1]))

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
        #elif key == ord("w"): hand_data.recalibrate_offset(hand_detector.get_details)
        elif key == ord("e"): print(hand_data.num_of_fingers_active())
