from ctypes import *
import win32service
import win32serviceutil
import win32api
import win32event
import win32evtlogutil
import os
import win32con
import win32gui

from win32con import WM_CLOSE
# from win32con import SWP_FRAMECHANGED 
# from win32con import SWP_NOMOVE 
# from win32con import SWP_NOSIZE 
# from win32con import SWP_NOZORDER
# from win32con import SW_HIDE
# from win32con import SW_FORCEMINIMIZE
# from win32con import SW_SHOWNORMAL
# from win32con import GW_OWNER 
# from win32con import GWL_STYLE 
# from win32con import GWL_EXSTYLE  
# from win32con import WS_CAPTION 
# from win32con import WS_EX_APPWINDOW 
# from win32con import WS_EX_CONTROLPARENT
# from win32con import WS_EX_TOOLWINDOW
# from win32con import WS_EX_WINDOWEDGE
# from win32con import WS_EX_LAYERED 
# from win32con import LWA_ALPHA
 

EnumWindows = windll.user32.EnumWindows
EnumWindowsProc = WINFUNCTYPE(c_bool, c_int, POINTER(c_int))
GetWindowText = windll.user32.GetWindowTextW
GetWindowTextLength = windll.user32.GetWindowTextLengthW
IsWindowVisible = windll.user32.IsWindowVisible
GetClassName = windll.user32.GetClassNameW
BringWindowToTop = windll.user32.BringWindowToTop
GetForegroundWindow = windll.user32.GetForegroundWindow
titles = []

def foreach_window(hwnd, lParam):
    if IsWindowVisible(hwnd):
        length = GetWindowTextLength(hwnd)
        classname = create_unicode_buffer(100 + 1)
        GetClassName(hwnd, classname, 100 + 1)
        buff = create_unicode_buffer(length + 1)
        GetWindowText(hwnd, buff, length + 1)
        if len(buff.value)!=0:
            titles.append( (hwnd, buff.value, classname.value, windll.user32.IsIconic(hwnd)) )
    return True

def refresh_wins():
    del titles[:]
    EnumWindows(EnumWindowsProc(foreach_window), 0)
    return titles

def listWindows(printed=False):
    newest_titles = refresh_wins()
    if printed:
        for t in newest_titles:
            print(type(t[1]))
    return newest_titles

def findWindow(windowName):
    # try to find window with exact title
    hwnd = win32gui.FindWindow(None, windowName)
    if hwnd != 0: return hwnd
    # try to find window with substring
    winds = listWindows()
    for w in winds:
        if windowName in w[1]:
            return w[0]
    # failed to find window
    raise Exception("findWindow() could not find [{}]".format(windowName))


# change window size
def changeWindow(hwnd, w, h):
    x, y, xl, yl = win32gui.GetWindowRect(hwnd)
    h = h+50 # add 50 for title bar
    win32gui.MoveWindow(hwnd, x, y, w, h, True)


# close window
def closeWindow(hwnd):   
    win32gui.PostMessage(hwnd,win32con.WM_CLOSE,0,0)
