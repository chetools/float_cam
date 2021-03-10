import PySimpleGUI as sg
from ctypes import c_bool, c_uint8, c_int, Structure, c_float

sg.theme('DarkBlack1')
sg.theme_slider_color('#303050')


def make_layout(valid_ids,dim):

    top_height_layout = [[sg.Text('V-Crop', font=('', 15), pad=((30, 0), (15, 0)))],
                         [sg.Slider(range=(100, 0), orientation='v', size=(16, 25), pad=((10, 0), (10, 25)), default_value=0, key='T', enable_events=True),
                          sg.Slider(range=(0, 100), orientation='v', size=(16, 25), pad=((10, 5), (10, 25)), default_value=0, key='B', enable_events=True)]
                         ]

    left_width_layout = [[
        sg.Column([[sg.Text('H-Crop', font=('', 15), pad=((30, 0), (8, 0)))]]), 
        sg.Column([[sg.Slider(range=(0, 100), orientation='h', size=(25, 20), pad=((10, 26), 10),
                                                                         default_value=0, key='L', enable_events=True)],
        [sg.Slider(range=(100, 0), orientation='h', size=(25, 20), pad=((10, 26), (0, 10)),
                                                                           default_value=0, key='R', enable_events=True)]])]]

    cam_flip_layout = [sg.Text('Camera', font=('', 18), pad=(5, (16, 27))),
                       sg.Spin(list(range(0, len(valid_ids))), initial_value=dim.ID, font=('', 20), pad=((0, 30), (18, 27)), key='ID',
                               background_color='#303050', text_color='#FFFFFF', enable_events=True),
                       sg.Checkbox(default=dim.hflip, text='H-Flip',  font=('', 18), pad=((0, 30), (18, 27)), key='hflip', enable_events=True)]

    chroma_layout = [
        [sg.Text('Hue LoPass', font=('', 15), pad=(10, (12, 0))),
         sg.Slider(range=(0, 255), orientation='h', size=(20, 20), default_value=dim.hue_loPass, key='hue_loPass', enable_events=True )],
        [sg.Text('Hue HiPass', font=('', 15), pad=(10, (12, 0))),
         sg.Slider(range=(0, 255), orientation='h', size=(20, 20), default_value=dim.hue_hiPass, key='hue_hiPass', enable_events=True)],
        [sg.Text('Sat LoPass', font=('', 15), pad=(10, (12, 0))),
         sg.Slider(range=(0, 255), orientation='h', size=(20, 20), default_value=dim.sat_loPass, key='sat_loPass', enable_events=True)],
        [sg.Text('Bright LoPass', font=('', 15), pad=(10, (12, 0))),
         sg.Slider(range=(0, 255), orientation='h', size=(20, 20), default_value=dim.bright_loPass, key='bright_loPass', enable_events=True)],
        [sg.Text('Rotate', font=('', 15), pad=(10, (12, 0))),
         sg.Slider(range=(0, 3), orientation='h', size=(20, 20), default_value=dim.rotate, key='rotate', enable_events=True)],
        [sg.Text('Float Scale', font=('', 15), pad=(10, (12, 0))),
         sg.Slider(range=(1, 24), orientation='h', size=(20, 20), default_value=dim.fscale, resolution=0.1, key='fscale', enable_events=True)],
        [sg.Text('Web Scale', font=('', 15), pad=(10, (12, 0))),
         sg.Slider(range=(1, 4), orientation='h', size=(20, 20), default_value=dim.wscale, resolution=0.01, key='wscale', enable_events=True)],
    ]

    chroma_camera_layout = [[sg.Column(chroma_layout, element_justification='right', pad=(20, 15))],
                            cam_flip_layout]

    layout = [[sg.Column(left_width_layout, element_justification='right')],
              [sg.Column(top_height_layout, element_justification='center'),
               sg.Column(chroma_camera_layout, element_justification='center')]
              ]

    return layout


class Dim(Structure):
    _fields_ = [('change', c_bool), ('ID', c_int), ('T', c_int), ('B', c_int), ('L', c_int), ('R', c_int), ('fscale', c_float), ('wscale', c_float),
                ('rotate', c_int), ('hflip', c_bool), ('hue_loPass',
                                                       c_int), ('hue_hiPass', c_int), ('sat_loPass', c_int),
                ('bright_loPass', c_int)]


def handle_key(event, value, dim,window):
    print(event.__dict__)
    print(window.current_location())
    # window.move(window.current_location()[0]+5,window.current_location()[1])
    dim.acquire()
    dim.change=True
    print(event.type=='ButtonPress')
    # if event.keysym=='Right' and dim.R<256:
        # dim.R+=1
    dim.release()

def config(dim, window2, terminate):

    while True:
        event, values = window2.Read(timeout=100)
        if event != sg.TIMEOUT_KEY:
            print(event, values)
            dim.acquire()
            dim.change = True
            if event == 'L':
                dim.L = int(values['L'])
            elif event == 'R':
                dim.R = int(values['R'])
            elif event == 'ID':
                dim.ID = int(values['ID'])
            elif event == 'T':
                dim.T = int(values['T'])
            elif event == 'B':
                dim.B = int(values['B'])
            elif event == 'rotate':
                dim.rotate = int(values['rotate'])
            elif event == 'hflip':
                dim.hflip = bool(values['hflip'])
            elif event == 'fscale':
                dim.fscale = values['fscale']
            elif event == 'wscale':
                dim.wscale = values['wscale']
            elif event == 'hue_loPass':
                dim.hue_loPass = int(values['hue_loPass'])
            elif event == 'hue_hiPass':
                dim.hue_hiPass = int(values['hue_hiPass'])
            elif event == 'sat_loPass':
                dim.sat_loPass = int(values['sat_loPass'])
            elif event == 'bright_loPass':
                dim.bright_loPass = int(values['bright_loPass'])
            dim.release()
        if event is None or event == sg.WIN_CLOSED or event == 'Exit':
            terminate.set()
            break


def dim_init(dim):
    dim.ID = 0
    dim.L = 0
    dim.R = 0
    dim.T = 0
    dim.B = 0
    dim.fscale = 2.
    dim.wscale = 2. 
    dim.rotate = 0
    dim.hflip = True
    dim.hue_loPass = 40
    dim.hue_hiPass = 90
    dim.sat_loPass = 90
    dim.bright_loPass = 20


