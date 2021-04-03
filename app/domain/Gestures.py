

class Gestures:
    def __init__(self):
        '''
        self.no_gesture = 0
        self.doing_other_things = 0
        self.stop_sign = 0
        self.thumb_up = 0
        self.sliding_down = 0
        self.sliding_up = 0
        self.swiping_right = 0
        self.swiping_left = 0
        self.turning_clockwise = 0
        '''
        self.label = ""

    def set_label(self, gesture):
        labels = ["No gesture", "Doing other things", "Stop Sign", "Thumb Up", "Sliding Two Fingers Down", 
            "Sliding Two Fingers Up", "Swiping Right", "Swiping Left", "Turning Hand Clockwise"]
        self.label = labels[gesture]
