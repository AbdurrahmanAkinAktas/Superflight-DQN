import win32gui
import win32ui
from ctypes import windll

from PIL import Image
import numpy as np
import cv2
import time
from windowFind import findWindow


# grabs current window image
def windowGrab(windowName, first, clientArea=True):
	hwnd = win32gui.FindWindow(None, windowName)

	if clientArea:
		left, top, right, bot = win32gui.GetClientRect(hwnd)
	else:	
		left, top, right, bot = win32gui.GetWindowRect(hwnd)
	w = right - left
	h = bot - top

	hwndDC = win32gui.GetWindowDC(hwnd)
	mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
	saveDC = mfcDC.CreateCompatibleDC()

	saveBitMap = win32ui.CreateBitmap()
	saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
	saveDC.SelectObject(saveBitMap)

	win32gui.SetForegroundWindow(hwnd)

	if clientArea:
		result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
	else:
		result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)

	bmpinfo = saveBitMap.GetInfo()
	bmpstr = saveBitMap.GetBitmapBits(True)

	img = Image.frombuffer(
		'RGB',
		(bmpinfo['bmWidth'], bmpinfo['bmHeight']),
		bmpstr, 'raw', 'BGRX', 0, 1)

	win32gui.DeleteObject(saveBitMap.GetHandle())
	saveDC.DeleteDC()
	mfcDC.DeleteDC()
	win32gui.ReleaseDC(hwnd, hwndDC)

	if result == 1:
		return(cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB))
		# img.save("grab.png")
		#print("saved window to grab.png")
	else:
		print("ERROR: could not capture", windowName)




# gets live capture of window 
# (clientArea is the innards of the window, no title bar)
def windowFeed(windowName, clientArea=True):
	# set up
	hwnd = findWindow(windowName)
	if clientArea:
		left, top, right, bot = win32gui.GetClientRect(hwnd)
	else:
		left, top, right, bot = win32gui.GetWindowRect(hwnd)
	w = right - left
	h = bot - top

	hwndDC = win32gui.GetWindowDC(hwnd)
	mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
	saveDC = mfcDC.CreateCompatibleDC()
	saveBitMap = win32ui.CreateBitmap()
	saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
	saveDC.SelectObject(saveBitMap)

	# cv2.namedWindow('capure',cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('capure', 400,300)

	# main loop
	while True:
		oldTime = time.time()
		if clientArea:
			result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 1)
		else:
			result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 0)
		bmpinfo = saveBitMap.GetInfo()
		bmpstr = saveBitMap.GetBitmapBits(True)

		img = Image.frombuffer(
			'RGB',
			(bmpinfo['bmWidth'], bmpinfo['bmHeight']),
			bmpstr,
			'raw',
			'RGBX', 0, 1)

		if result == 1:
			cv2.imshow('capure', np.array(img))
			pass
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		print("FPS: {:.2f}".format(1/(time.time()-oldTime)))


	# clean up
	win32gui.DeleteObject(saveBitMap.GetHandle())
	saveDC.DeleteDC()
	mfcDC.DeleteDC()
	win32gui.ReleaseDC(hwnd, hwndDC)


