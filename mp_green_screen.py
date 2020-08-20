import time, os, configparser, argparse
from sys import platform
from itertools import cycle
import multiprocessing as mp
import cv2
import PySimpleGUI as sg
import numpy as np
from ctypes import c_bool, c_uint8, c_int, Structure, c_float
BUFFER_SIZE = 1280*960*3

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ini', type=str)
parser.add_argument('-f', '--find_cam', help='cycle through cameras', action='store_true')
args = parser.parse_args()

config_file = args.ini if args.ini else 'config.ini'
config = configparser.ConfigParser()

def find_cam(frame, new_frame, dim):

    ids = cycle(range(6))
    while True:
        id = next(ids)
        cam = cv2.VideoCapture(id)
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

    while True:
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
        time.sleep(0.1)

class Dim(Structure):
    _fields_ = [('ID', c_int), ('T', c_int), ('B', c_int), ('L', c_int), ('R', c_int), ('scale', c_float),
                ('rotate', c_int), ('hue_loPass', c_int), ('hue_hiPass', c_int), ('sat_loPass', c_int),
                ('bright_loPass', c_int)]

def read_config(dim):
    config.read(config_file)
    dim.acquire()
    dim.ID = int(config['Default']['Cam_ID'])
    dim.L = int(config['Default']['Left'])
    dim.R = int(config['Default']['Right'])
    dim.B = int(config['Default']['Bottom'])
    dim.T = int(config['Default']['Top'])
    dim.scale = float(config['Default']['Scale'])
    dim.rotate = int(config['Default']['Rotate'])
    dim.hue_loPass = int(config['Default']['Hue_LowPass'])
    dim.hue_hiPass = int(config['Default']['Hue_HighPass'])
    dim.sat_loPass = int(config['Default']['Sat_LowPass'])
    dim.bright_loPass = int(config['Default']['Bright_LowPass'])
    dim.release()

def monitor_config_change(dim):
    old_config_time = os.stat(config_file)[8]
    while True:
        new_config_time = os.stat(config_file)[8]
        if new_config_time != old_config_time:
            read_config(dim)
        time.sleep(1)


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
    dim = mp.Value(Dim)
    read_config(dim)



    monitor_config_proc = ctx.Process(target=monitor_config_change, args=(dim,), daemon=True)
    monitor_config_proc.start()

    if args.find_cam:
        find_cam_proc = ctx.Process(target=find_cam, args=(frame, new_frame, dim), daemon=True)
        find_cam_proc.start()
    else:
        update_frame_proc = ctx.Process(target=update_frame, args=(frame, new_frame, dim), daemon=True)
        update_frame_proc.start()


    while True:
        event, values = window.Read(timeout=10)

        if new_frame.value:
            frame_array = np.frombuffer(frame,dtype=c_uint8)
            window.FindElement('image').Update(data=frame_array.tobytes())
            new_frame.value=False
