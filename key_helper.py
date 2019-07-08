from win32con import KEYEVENTF_KEYUP
from win32api import keybd_event
import time

UP = 0x53
DOWN = 0x57
LEFT = 0x41
RIGHT = 0x44
SPACE = 0x20
NOTHING = 0x0
ESC = 0x1B



def press(*args):
    '''
    one press, one release.
    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
    '''
    for i in args:
        if i != NOTHING:
            keybd_event(i, 0,0,0)

    time.sleep(.01)

    for i in args:
        if i != NOTHING:
            keybd_event(i,0 ,KEYEVENTF_KEYUP ,0)
