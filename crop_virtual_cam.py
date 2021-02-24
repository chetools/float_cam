import time
from sys import platform
import multiprocessing as mp
import cv2
import PySimpleGUI as sg
import numpy as np
from ctypes import c_bool, c_uint8, c_int, Structure, c_float
from layout import make_layout, config, dim_init, Dim
import pyvirtualcam
import signal
CAP_W = 1920
CAP_H = 1080
CAP_BUFFER_SIZE = CAP_W*CAP_H*3
VC_W = 1280
VC_H = 720
VC_BUFFER_SIZE = VC_W*VC_H*4


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

def send_vc_frame(vc_frame_buffer, vc_frame_ready):
    vc_frame = np.frombuffer(vc_frame_buffer, dtype=c_uint8).reshape((VC_H, VC_W, 4))
    vc_copy = np.copy(vc_frame)
    with pyvirtualcam.Camera(width=VC_W, height=VC_H, fps=30) as vc:
        while True:
            if vc_frame_ready.is_set():
                vc.send(vc_frame)
                vc_copy[:,:] = vc_frame
                vc_frame_ready.clear()
            else:
                vc.send(vc_copy)
            # vc.sleep_until_next_frame()
            
def lrtbhw(dim, cam):
    w = int(cam.get(3))
    h = int(cam.get(4))
    L, T, R, B = dim.L, dim.T, dim.R, dim.B

    r = int(w*L/100)
    l = int(w*R/100)
    t = int(h*T/100)
    b = int(h*B/100)
    scale = dim.scale
    return l,r,t,b, scale

def update_frames(frame_buffer, new_frame, vc_frame_buffer, vc_frame_ready, dim, valid_ids):
    print(f'update frame valid: {valid_ids}')
    old_ID = dim.ID
    cam = cv2.VideoCapture(valid_ids[dim.ID], cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_H)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_W)
    l,r,t,b,scale=lrtbhw(dim,cam)
    transparent = np.array([0, 255, 0], dtype=c_uint8)
    frame = np.frombuffer(frame_buffer, dtype=c_uint8)
    vc_frame = np.frombuffer(vc_frame_buffer, dtype=c_uint8).reshape((VC_H, VC_W, 4))

    while True:
        try:
            ret, img = cam.read()
            dim.acquire()
            if dim.change:
                if dim.ID != old_ID:
                    cam = cv2.VideoCapture(valid_ids[dim.ID], cv2.CAP_DSHOW)
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
                    old_ID = dim.ID
                if not cam:
                    raise EnvironmentError
                l,r,t,b,scale=lrtbhw(dim,cam)
                dim.change = False
            img = np.rot90(img, dim.rotate)            
            if dim.hflip:
                img = np.fliplr(img)
            h,w,_ = img.shape
            img = cv2.resize(img[t:(h-b), l:(w-r), :], None,
                            fx=1/scale,fy=1/scale,
                            interpolation=cv2.INTER_CUBIC)

            dim.release()
            # data = cv2.imencode('.png', img )[1][:, 0]
            # frame[:data.shape[0]] = data
            # new_frame.set()

            if not vc_frame_ready.is_set():
                vc_frame.fill(0)     
                img_h, img_w,_ = img.shape
          
                if img_h > VC_H and img_w > VC_W:
                    vc_frame[:, :, :3] = cv2.cvtColor(img[(img_h//2 - VC_H//2):(img_h//2 + (VC_H - VC_H//2)), 
                        (img_w//2 - VC_W//2):(img_w//2 + (VC_W - VC_W//2))], cv2.COLOR_BGR2RGB)

                elif img_h > VC_H and img_w <= VC_W:
                    vc_frame[:,(VC_W//2-img_w//2):(VC_W//2+(img_w - img_w//2)), :3] = cv2.cvtColor(
                        img[(img_h//2 - VC_H//2):(img_h//2 + (VC_H - VC_H//2)), :], cv2.COLOR_BGR2RGB)

                elif img_h <= VC_H and img_w > VC_W:
                    vc_frame[(VC_H//2 - img_h//2):(VC_H//2+(img_h-img_h//2)), :, :3] = cv2.cvtColor(
                        img[:, (img_w//2 - VC_W//2):(img_w//2 + (VC_W - VC_W//2))], cv2.COLOR_BGR2RGB)

                elif img_h <= VC_H and img_w <= VC_W:
                    vc_frame[(VC_H//2 - img_h//2):(VC_H//2+(img_h-img_h//2)),(VC_W//2-img_w//2):(VC_W//2+(img_w - img_w//2)), :3] = cv2.cvtColor(
                        img, cv2.COLOR_BGR2RGB)

                vc_frame_ready.set()

        except:
            dim.change = False
            dim.release()
            raise EnvironmentError
            pass
        # time.sleep(0.03)


if __name__ == '__main__':


    if platform == 'win32':
        window = sg.Window('FloatCam', [[sg.Image(filename='', key='image', background_color='#00FF00')], ],
                           transparent_color='#00FF00', no_titlebar=True, grab_anywhere=True, return_keyboard_events=True,
                           keep_on_top=True, force_toplevel=True, element_padding=(0, 0), margins=(0, 0))
    else:
        window = sg.Window('FloatCam', [[sg.Image(filename='', key='image', background_color='black')], ],
                           no_titlebar=True, grab_anywhere=True, return_keyboard_events=True,
                           keep_on_top=True, force_toplevel=True, element_padding=(0, 0), margins=(0, 0))

    ctx = mp.get_context('spawn')
    frame_buffer = mp.RawArray(c_uint8, CAP_BUFFER_SIZE)
    new_frame = ctx.Event()

    vc_frame_buffer = mp.RawArray(c_uint8, VC_BUFFER_SIZE)
    vc_frame_ready = ctx.Event()

    terminate = ctx.Event()
    dim = mp.Value(Dim)
    valid_ids = find_cam()
    dim_init(dim)

    window2 = sg.Window('FloatCam Controls', make_layout(valid_ids))

    config_proc = ctx.Process(target=config, args=(
        dim, window2, terminate), daemon=True)
    config_proc.start()

    update_frame_proc = ctx.Process(target=update_frames, args=(
        frame_buffer, new_frame, vc_frame_buffer, vc_frame_ready, dim, valid_ids), daemon=True)
    update_frame_proc.start()

    send_vc_frame_proc = (ctx.Process(target=send_vc_frame, args=(
        vc_frame_buffer, vc_frame_ready), daemon=True))
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

        if new_frame.is_set():
            frame_array = np.frombuffer(frame_buffer, dtype=c_uint8)
            window['image'].Update(data=frame_array.tobytes())
            new_frame.clear()

    



