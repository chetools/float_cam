import time, os, configparser, argparse
from sys import platform
from itertools import cycle
import multiprocessing as mp
import cv2
import PySimpleGUI as sg
import numpy as np
from ctypes import c_bool, c_uint8, c_int, Structure, c_float
BUFFER_SIZE = 1920*1080*3

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ini', type=str)
args = parser.parse_args()

config_file = args.ini if args.ini else 'config.ini'
config = configparser.ConfigParser()

def find_cam(frame, new_frame, dim):

    ids = cycle(range(6))
    while True:
        id = next(ids)
        cam = cv2.VideoCapture(3)
        try:
            ret, img = cam.read()
            dim.acquire()
            hsv = cv2.cvtColor(np.fliplr(img)[dim.T:-dim.B, dim.L:-dim.R,:],cv2.COLOR_BGR2HSV)
            mask=np.zeros((hsv.shape[0],hsv.shape[1]))
            np.logical_or(np.less(hsv[:,:,0],dim.hue_loPass), mask, out=mask)
            np.logical_or(np.greater(hsv[:,:,0],dim.hue_hiPass), mask, out=mask)
            np.logical_or(np.less(hsv[:,:,1], dim.sat_loPass),mask, out=mask)
            np.logical_or(np.less(hsv[:,:,2], dim.bright_loPass),mask, out=mask)

            frame_array = np.frombuffer(frame,dtype=c_uint8)
            img_resize = cv2.resize(mask[:,:,None]*np.fliplr(img)[dim.T:-dim.B, dim.L:-dim.R,:],
                                    dsize=(int((img.shape[1]-dim.L-dim.R)/dim.scale),
                                           int((img.shape[0]-dim.T-dim.B)/dim.scale)),
                                    interpolation=cv2.INTER_CUBIC)
            data = cv2.imencode('.png', np.rot90(img_resize, dim.rotate))[1][:,0]
            dim.release()
            frame_array[:data.shape[0]] = data
            new_frame.value=True
            print(f'VALID ID: {id}', sep='')
            time.sleep(1)
        except:
            print(f'Invalid camera ID: {id}', sep='')
        print()


def update_frame(frame, new_frame, dim):

    cam = cv2.VideoCapture(dim.ID)
    scale=2
    w = cam.get(3)
    h = cam.get(4)
    l = int(w*dim.L/100)
    r = int(l + (w-l)*dim.W/100)
    t = int(h*dim.T/100)
    b = int(t + (h-t)*dim.H/100)

    while True:
        try:
            ret, img = cam.read()
            dim.acquire()
            if dim.change:
                cam = cv2.VideoCapture(dim.ID)
                print(cam)
                if not cam:
                    raise EnvironmentError
                w = cam.get(3)
                h = cam.get(4)
                l = int(w*dim.L/100)
                r = int(l + (w-l)*dim.W/100)
                t = int(h*dim.T/100)
                b = int(t + (h-t)*dim.H/100)
                scale=dim.Scale
                dim.change=False
            hsv = cv2.cvtColor(np.fliplr(img)[t:b, l:r,:],cv2.COLOR_BGR2HSV)
            mask=np.zeros((hsv.shape[0],hsv.shape[1]))
            np.logical_or(np.less(hsv[:,:,0],dim.hue_loPass), mask, out=mask)
            np.logical_or(np.greater(hsv[:,:,0],dim.hue_hiPass), mask, out=mask)
            np.logical_or(np.less(hsv[:,:,1], dim.sat_loPass),mask, out=mask)
            np.logical_or(np.less(hsv[:,:,2], dim.bright_loPass),mask, out=mask)

            frame_array = np.frombuffer(frame,dtype=c_uint8)
            img_resize = cv2.resize(mask[:,:,None]*np.fliplr(img)[t:b, l:r,:],
                                    dsize=(int((r-l)/scale),
                                           int((b-t)/scale)),
                                    interpolation=cv2.INTER_CUBIC)
            data = cv2.imencode('.png', np.rot90(img_resize, 0))[1][:,0]
            dim.release()
            frame_array[:data.shape[0]] = data
            new_frame.value=True
        except:
            pass
        time.sleep(0.1)

class Dim(Structure):
    _fields_ = [('change', c_bool), ('ID', c_int), ('T', c_int), ('H', c_int), ('L', c_int), ('W', c_int), ('Scale', c_float),
                ('rotate', c_int), ('hue_loPass', c_int), ('hue_hiPass', c_int), ('sat_loPass', c_int),
                ('bright_loPass', c_int)]

def config(dim, terminate):
    layout = [[
        sg.Slider(range=(1, 100), orientation='h', size=(50, 20), default_value=0, key='L',enable_events=True)],
        [sg.Slider(range=(1, 100), orientation='h', size=(50, 20), default_value=100, key='W',enable_events=True)],
        [sg.Slider(range=(0, 10), orientation='v', size=(10, 15), default_value=0, key='ID',enable_events=True),
         sg.Slider(range=(1, 100), orientation='v', size=(10, 15), default_value=0, key='T',enable_events=True),
         sg.Slider(range=(1, 100), orientation='v', size=(10, 15), default_value=0, key='H',enable_events=True),
         sg.Slider(range=(1, 4), orientation='v', size=(10, 15), resolution = 0.25,
                   default_value=2, key='Scale',enable_events=True),
         sg.Slider(range=(1, 100), orientation='v', size=(10, 15), default_value=40, key='hue_loPass',enable_events=True),
         sg.Slider(range=(1, 100), orientation='v', size=(10, 15), default_value=90, key='hue_hiPass',enable_events=True),
         sg.Slider(range=(1, 100), orientation='v', size=(10, 15), default_value=50, key='sat_loPass',enable_events=True),
         sg.Slider(range=(1, 100), orientation='v', size=(10, 15), default_value=50, key='bright_loPass',enable_events=True),
         ]
    ]
    window2 = sg.Window('FloatCam Controls', layout)


    while True:
        event, values = window2.Read(timeout=100)
        if event != sg.TIMEOUT_KEY:
            dim.acquire()
            dim.change=True
            if event=='L':
                dim.L = int(values['L'])
            elif event=='W':
                dim.W = int(values['W'])
            elif event=='ID':
                dim.ID = int(values['ID'])
            elif event=='T':
                dim.T = int(values['T'])
            elif event=='H':
                dim.H = 100-int(values['H'])
            elif event=='Scale':
                dim.Scale=values['Scale']
            elif event=='hue_loPass':
                dim.hue_loPass=int(values['hue_loPass'])
            elif event=='hue_hiPass':
                dim.hue_hiPass=int(values['hue_hiPass'])
            elif event=='sat_loPass':
                dim.sat_loPass=int(values['sat_loPass'])
            elif event=='bright_loPass':
                dim.bright_loPass=int(values['bright_loPass'])
            dim.release()
        if event is None or event == sg.WIN_CLOSED or event == 'Exit':
            break

    terminate.value=True

def dim_init(dim):
    dim.ID=3
    dim.L=0
    dim.W=100
    dim.T=0
    dim.H=100
    dim.Scale=2
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
    dim_init(dim)


    config_proc = ctx.Process(target=config, args=(dim, terminate), daemon=True)
    config_proc.start()

    update_frame_proc = ctx.Process(target=update_frame, args=(frame, new_frame, dim), daemon=True)
    update_frame_proc.start()


    while True:
        event, values = window.Read(timeout=10)
        if event is None or event == sg.WIN_CLOSED or event == 'Exit' or terminate.value:
            break

        if new_frame.value:
            frame_array = np.frombuffer(frame,dtype=c_uint8)
            window.FindElement('image').Update(data=frame_array.tobytes())
            new_frame.value=False