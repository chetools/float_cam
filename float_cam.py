import time
from sys import platform
import multiprocessing as mp
import cv2
import PySimpleGUI as sg
import numpy as np
from ctypes import c_bool, c_uint8, c_int, Structure, c_float
from layout import make_layout
BUFFER_SIZE = 1920*1080*3

def find_cam():
    valid_id=[]
    for i in range(8):
        cam = cv2.VideoCapture(i)
        try:
            ret, img = cam.read()
            if img is not None:
                valid_id.append(i)
        except:
            pass
    print('valid', valid_id)
    return(valid_id)


def update_frame(frame, new_frame, dim, valid_ids):
    print(f'update frame valid: {valid_ids}')
    old_ID = dim.ID
    cam = cv2.VideoCapture(valid_ids[dim.ID])
    scale=2
    w = cam.get(3)
    h = cam.get(4)
    H = 100-dim.H
    l = int(w*dim.L/100)
    r = int(l + (w-l)*dim.W/100)
    t = int(h*dim.T/100)
    b = int(t + (h-t)*H/100)

    while True:
        try:
            ret, img = cam.read()
            dim.acquire()
            if dim.change:
                if dim.ID != old_ID:
                    cam = cv2.VideoCapture(valid_ids[dim.ID])
                    old_ID = dim.ID
                if not cam:
                    raise EnvironmentError
                w = cam.get(3)
                h = cam.get(4)
                l = int(w*dim.L/100)
                r = int(l + (w-l)*dim.W/100)
                t = int(h*dim.T/100)
                b = int(t + (h-t)*H/100)
                scale=dim.scale
                dim.change=False
            hsv = cv2.cvtColor(np.fliplr(img)[t:b, l:r,:],cv2.COLOR_BGR2HSV)
            mask=np.zeros((hsv.shape[0],hsv.shape[1]))
            np.logical_or(np.less(hsv[:,:,0],dim.hue_loPass), mask, out=mask)
            np.logical_or(np.greater(hsv[:,:,0],dim.hue_hiPass), mask, out=mask)
            np.logical_or(np.less(hsv[:,:,1], dim.sat_loPass),mask, out=mask)
            np.logical_or(np.less(hsv[:,:,2], dim.bright_loPass),mask, out=mask)

            frame_array = np.frombuffer(frame,dtype=c_uint8)
            if dim.hflip:
                img=np.fliplr(img)

            img_resize = cv2.resize(mask[:,:,None]*img[t:b, l:r,:],
                                    dsize=(int((r-l)/scale),
                                           int((b-t)/scale)),
                                    interpolation=cv2.INTER_CUBIC)
            data = cv2.imencode('.png', np.rot90(img_resize, dim.rotate))[1][:,0]
            dim.release()
            frame_array[:data.shape[0]] = data
            new_frame.value=True
        except:
            dim.change=False
            dim.release()
            pass
        time.sleep(0.03)

class Dim(Structure):
    _fields_ = [('change', c_bool), ('ID', c_int), ('T', c_int), ('H', c_int), ('L', c_int), ('W', c_int), ('scale', c_float),
                ('rotate', c_int), ('hflip', c_bool), ('hue_loPass', c_int), ('hue_hiPass', c_int), ('sat_loPass', c_int),
                ('bright_loPass', c_int)]

def config(dim, window2, terminate):

    while True:
        event, values = window2.Read(timeout=100)
        if event != sg.TIMEOUT_KEY:
            print(event, values)
            dim.acquire()
            dim.change=True

            if event=='hflip':
                dim.hflip = bool(values['hflip'])
            elif event=='scale':
                dim.scale=values['scale']
            else:
                setattr(dim,event,int(values[event]))
            dim.release()
        if event is None or event == sg.WIN_CLOSED or event == 'Exit':
            terminate.value=True
            break



def dim_init(dim):
    dim.ID=0
    dim.L=0
    dim.W=100
    dim.T=0
    dim.H=100
    dim.scale=2.
    dim.rotate=0
    dim.hflip=True
    dim.hue_loPass = 40
    dim.hue_hiPass = 90
    dim.sat_loPass = 50
    dim.bright_loPass =50

if __name__ == '__main__':

    if platform =='win32':
        window = sg.Window('Demo Application - OpenCV Integration', [[sg.Image(filename='', key='image', background_color='black')],],
                           transparent_color='black', no_titlebar=True, grab_anywhere=True, return_keyboard_events=True,
                           keep_on_top=True, force_toplevel=True, element_padding=(0,0),margins=(0,0) )
    else:
        window = sg.Window('Demo Application - OpenCV Integration', [[sg.Image(filename='', key='image', background_color='black')],],
                           no_titlebar=True, grab_anywhere=True, return_keyboard_events=True,
                           keep_on_top=True, force_toplevel=True, element_padding=(0,0),margins=(0,0) )



    ctx = mp.get_context('spawn')
    frame = mp.RawArray(c_uint8, BUFFER_SIZE)
    new_frame = ctx.Value(c_bool, False)
    terminate = ctx.Value(c_bool, False)
    dim = mp.Value(Dim)
    valid_ids=find_cam()
    # valid_ids=[1,2,3]
    # print(f'valid: {valid_ids}')
    dim_init(dim)


    window2 = sg.Window('FloatCam Controls', make_layout(valid_ids))

    config_proc = ctx.Process(target=config, args=(dim, window2, terminate), daemon=True)
    config_proc.start()

    update_frame_proc = ctx.Process(target=update_frame, args=(frame, new_frame, dim, valid_ids), daemon=True)
    update_frame_proc.start()


    while True:
        event, values = window.Read(timeout=10)
        if event is None or event == sg.WIN_CLOSED or event == 'Exit' or terminate.value:
            break

        if new_frame.value:
            frame_array = np.frombuffer(frame,dtype=c_uint8)
            window.FindElement('image').Update(data=frame_array.tobytes())
            new_frame.value=False