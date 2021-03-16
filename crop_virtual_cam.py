import time
from sys import platform
import multiprocessing as mp
import cv2
import PySimpleGUI as sg
import numpy as np
from ctypes import c_bool, c_uint8, c_int, Structure, c_float
from layout import make_layout, config, dim_init, Dim, handle_key
import pyvirtualcam
import signal

import glob
from pathlib import Path
import re


CAP_W = 1920
CAP_H = 1080
CAP_BUFFER_SIZE = CAP_W*CAP_H*3
VC_W = 1280
VC_H = 720
VC_BUFFER_SIZE = VC_W*VC_H*4*2
kernel = np.ones((3,3),np.uint8)
border = np.array([50,50,50],dtype=c_uint8)


def find_cam():
    valid_id = []
    for i in range(8):
        cam = cv2.VideoCapture(i)
        try:
            ret, img = cam.read()
            if img is not None:
                valid_id.append(i)
        except:
            pass
    return(valid_id)

def send_vc_frame(vc_frame_buffer, vc_frame0):
    vc_frame = np.frombuffer(vc_frame_buffer, dtype=c_uint8).reshape((2, VC_H, VC_W, 4))
    with pyvirtualcam.Camera(width=VC_W, height=VC_H, fps=30) as vc:
        while True:
            if vc_frame0.is_set():
                vc.send(vc_frame[0])
            else:
                vc.send(vc_frame[1])

def read_background_images():
    pat=re.compile(r'(\d+)')

    files = glob.glob('background/chemical_plant/*.png')
    num = np.argsort(np.array([int(pat.search(afile)[1]) for afile in files]))
    num_background_images=len(files)
    background_images = np.zeros((num_background_images,720,1280,3))

    for i,afile in enumerate(files):
        background_images[num[i]]=cv2.cvtColor(cv2.imread(afile),cv2.COLOR_BGR2RGB)
    return num_background_images, background_images

def lrtbhw(dim, cam):
    w = int(cam.get(3))
    h = int(cam.get(4))
    L, T, R, B = dim.L, dim.T, dim.R, dim.B

    l = int(w*L/100)
    r = int(w*R/100)
    t = int(h*T/100)
    b = int(h*B/100)
    fscale = dim.fscale
    wscale = dim.wscale
    scale=min(fscale,wscale)
    if dim.rotate==1 or dim.rotate==3:
        h,w=w,h

    scale=min(fscale,wscale)
    hh=h-b-t
    ww=w-r-l
    maskL=np.zeros((int((hh)/scale), int((ww)/scale)), dtype=c_uint8)
    maskL_float = np.zeros((int((hh)/scale), int((ww)/scale)), dtype=c_float)
    maskL_smooth = np.zeros_like(maskL_float)
    imgL=np.zeros((int((hh)/scale), int((ww)/scale),3), dtype=c_uint8)
    hsv=np.zeros_like(imgL)
    y,x=np.indices((int((hh)/scale),int((ww)/scale)))
    h2,w2= hh/scale/2, ww/scale/2
    rad=min(h2,w2)
    circle_idx=np.sqrt((x-w2)**2 + (y-h2)**2)<rad
    circle_mask=np.zeros_like(maskL)
    circle_mask[circle_idx]=1

    scale=max(fscale,wscale)
    imgS=np.zeros((int((hh)/scale), int((ww)/scale),3), dtype=c_uint8)
    maskS_float = np.zeros(imgS.shape[:2], dtype=c_float)

    # indices for floating circle border
    y,x=np.indices((int((hh)/fscale),int((ww)/fscale)))
    h2,w2= hh/fscale/2, ww/fscale/2
    rad=min(h2,w2)
    fcircle_idx=np.abs(np.sqrt((x-w2)**2 + (y-h2)**2)-rad)<=1 

    return (l,r,t,b, h,w, fscale, wscale, maskL, maskL_float, maskL_smooth, maskS_float, circle_mask, 
        fcircle_idx, imgL, imgS, hsv)

def update_frames(frame_buffer, new_frame, vc_frame_buffer, vc_frame0, dim, valid_ids):
    print(f'update frame valid: {valid_ids}')
    old_ID = dim.ID
    cam = cv2.VideoCapture(valid_ids[dim.ID])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_H)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_W)
    (l,r,t,b,h,w,fscale,wscale, maskL, maskL_float, maskL_smooth, maskS_float, circle_mask, fcircle_idx, imgL,
        imgS, hsv)=lrtbhw(dim,cam)
    transparent = np.array([0, 255, 0], dtype=c_uint8)
    num_background, background_images = read_background_images()

    frame = np.frombuffer(frame_buffer, dtype=c_uint8)
    vc_frame = np.frombuffer(vc_frame_buffer, dtype=c_uint8).reshape((2, VC_H, VC_W, 4))
    tstart = time.time()
    i_background = 0
    background=background_images[0]
    while True:
        try:
            ret, img = cam.read()
            dim.acquire()
            if dim.change:
                if dim.ID != old_ID:
                    cam = cv2.VideoCapture(valid_ids[dim.ID])
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
                    ret, img = cam.read()
                    old_ID = dim.ID
                if not cam:
                    raise EnvironmentError
                tstart = time.time()
                i_background = 0
                background=background_images[0]

            ctime = time.time()


            (l,r,t,b,h,w,fscale,wscale, maskL, maskL_float, maskL_smooth, maskS_float, circle_mask, fcircle_idx,
                imgL, imgS, hsv)=lrtbhw(dim,cam)
            dim.change = False
            img = np.rot90(img, dim.rotate)            
            if dim.hflip:
                img = np.fliplr(img)
            dim.release()
            img=img[t:(h-b), l:(w-r), :]
            cv2.resize(img, dsize=(imgL.shape[1],imgL.shape[0]), dst=imgL,
                            interpolation=cv2.INTER_CUBIC)
            cv2.resize(img, dsize=(imgS.shape[1],imgS.shape[0]), dst=imgS,
                            interpolation=cv2.INTER_CUBIC)
            cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV,dst=hsv)
            maskL.fill(0)
            np.logical_or(
                np.less(hsv[:, :, 0], dim.hue_loPass), maskL, out=maskL)
            np.logical_or(np.greater(
                hsv[:, :, 0], dim.hue_hiPass), maskL, out=maskL)
            np.logical_or(
                np.less(hsv[:, :, 1], dim.sat_loPass), maskL, out=maskL)
            np.logical_or(
                np.less(hsv[:, :, 2], dim.bright_loPass), maskL, out=maskL)
            cv2.erode(maskL.astype(c_float), iterations=5, kernel=kernel, dst=maskL_float)
            cv2.dilate(maskL_float, iterations=5, kernel=kernel, dst=maskL_float)

            maskL.fill(0)
            contours, heirarchy = cv2.findContours(maskL_float.astype(c_uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            main_contour=np.argmax(np.array([cv2.contourArea(contour) for contour in contours]))
            child_contours=[j for j,heir in enumerate(heirarchy[0]) if (heir[3]==main_contour and
                cv2.contourArea(contours[j])>20.0)]

            cv2.drawContours(maskL, contours, main_contour, (1,1,1), -1, cv2.LINE_8)
            for child_contour in child_contours:
                cv2.drawContours(maskL, contours, child_contour, (0,0,0), -1, cv2.LINE_8)
            np.multiply(maskL, circle_mask, out=maskL)
            cv2.erode(maskL.astype(c_float), iterations=2, kernel=kernel, dst=maskL_float)
            cv2.resize(maskL_float, dsize=(imgS.shape[1],imgS.shape[0]), dst=maskS_float,
                            interpolation=cv2.INTER_CUBIC)

            imgLL=np.where(maskL_float[:,:,None].astype(c_bool), imgL, transparent[None, None, :])
            imgSS=np.where(maskS_float[:, :,None].astype(c_bool), imgS, transparent[None, None, :])
            if fscale<=wscale:
                img=imgLL
            else:
                img=imgSS
            img[fcircle_idx]=border
            data = cv2.imencode('.png',img)[1][:, 0]
            frame_array = np.frombuffer(frame, dtype=c_uint8)
            frame_array[:data.shape[0]] = data
            new_frame.set()

            idx=0
            if dim.wscale<=dim.fscale:
                img = imgLL
            else:
                img = imgSS
            img_h, img_w,_ = img.shape
            if vc_frame0.is_set():
                idx=1
          
            if (ctime-tstart)*1000 > 25:
                tstart=ctime
                i_background+=1
                background=background_images[i_background % num_background]
                vc_frame[idx,:,:,:3]=background.copy()

            if img_h > VC_H and img_w > VC_W:
                vc_frame[idx,:, :, :3] = cv2.cvtColor(img[(img_h//2 - VC_H//2):(img_h//2 + (VC_H - VC_H//2)), 
                    (img_w//2 - VC_W//2):(img_w//2 + (VC_W - VC_W//2))], cv2.COLOR_BGR2RGB)

            elif img_h > VC_H and img_w <= VC_W:
                vc_frame[idx,:,(VC_W//2-img_w//2):(VC_W//2+(img_w - img_w//2)), :3] = cv2.cvtColor(
                    img[(img_h//2 - VC_H//2):(img_h//2 + (VC_H - VC_H//2)), :], cv2.COLOR_BGR2RGB)

            elif img_h <= VC_H and img_w > VC_W:
                vc_frame[idx,(VC_H//2 - img_h//2):(VC_H//2+(img_h-img_h//2)), :, :3] = cv2.cvtColor(
                    img[:, (img_w//2 - VC_W//2):(img_w//2 + (VC_W - VC_W//2))], cv2.COLOR_BGR2RGB)

            elif img_h <= VC_H and img_w <= VC_W:
                vc_frame[idx,(VC_H//2 - img_h//2):(VC_H//2+(img_h-img_h//2)),(VC_W//2-img_w//2):(VC_W//2+(img_w - img_w//2)), :3] = cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB)
            vc_frame[idx,:,:,:3]=np.where(vc_frame[idx,:,:,:3]==transparent, background, vc_frame[idx,:,:,:3])
            if vc_frame0.is_set():
                vc_frame0.clear()
            else:
                vc_frame0.set()

        except:
            dim.change = False
            dim.release()
            raise EnvironmentError
            pass
        # time.sleep(0.03)


if __name__ == '__main__':


    if platform == 'win32':
        window = sg.Window('FloatCam', [[sg.Image(filename='', key='image', background_color='#00FF00')], ],
                           transparent_color='#00FF00', no_titlebar=True, grab_anywhere=True,
                           keep_on_top=True, force_toplevel=True, element_padding=(0, 0), margins=(0, 0), finalize=True)
    else:
        window = sg.Window('FloatCam', [[sg.Image(filename='', key='image', background_color='black')], ],
                           no_titlebar=True, grab_anywhere=True,
                           keep_on_top=True, force_toplevel=True, element_padding=(0, 0), margins=(0, 0), finalize=True)

    root = window.TKroot
    root.bind('<KeyPress>', lambda event, key='Press': window.write_event_value(event,key))
    root.bind('<MouseWheel>', lambda event, key='Wheel': window.write_event_value(event,key))
    root.bind('<Button-3>', lambda event, key='B3': window.write_event_value(event,key))
    root.bind('<B3-Motion>', lambda event, key='BM3': window.write_event_value(event,key))
    root.bind('<ButtonRelease-3>', lambda event, key='BR3': window.write_event_value(event,key))

    ctx = mp.get_context('spawn')
    frame_buffer = mp.RawArray(c_uint8, CAP_BUFFER_SIZE)
    new_frame = ctx.Event()

    vc_frame_buffer = mp.RawArray(c_uint8, VC_BUFFER_SIZE)
    vc_frame0 = ctx.Event()

    terminate = ctx.Event()
    dim = mp.Value(Dim)
    valid_ids = find_cam()
    dim_init(dim)

    window2 = sg.Window('FloatCam Controls', make_layout(valid_ids, dim))

    config_proc = ctx.Process(target=config, args=(
        dim, window2, terminate), daemon=True)
    config_proc.start()

    update_frame_proc = ctx.Process(target=update_frames, args=(
        frame_buffer, new_frame, vc_frame_buffer, vc_frame0, dim, valid_ids), daemon=True)
    update_frame_proc.start()

    send_vc_frame_proc = (ctx.Process(target=send_vc_frame, args=(
        vc_frame_buffer, vc_frame0), daemon=True))
    send_vc_frame_proc.start()

    def keyboard_interrupt_handler(signal, frame):
        window.close()
        window2.close()
        config_proc.stop()
        update_frame_proc.stop()
        send_vc_frame_proc.stop()
        exit()

    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    while True:
        event, values = window.Read(timeout=10)
        if event is None or event == sg.WIN_CLOSED or event == 'Exit' or terminate.is_set():
            break
        if event != sg.TIMEOUT_KEY:
            handle_key(event, values, dim, window)
        if new_frame.is_set():
            frame_array = np.frombuffer(frame_buffer, dtype=c_uint8)
            window['image'].Update(data=frame_array.tobytes())
            new_frame.clear()

    



