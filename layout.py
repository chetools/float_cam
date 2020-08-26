import PySimpleGUI as sg

sg.theme('DarkBlack1')
sg.theme_slider_color('#303050')


def make_layout(valid_ids):

    top_height_layout = [[sg.Text('Top', font=('',15),pad=((32,0),(15,0))), sg.Text('Height', font=('',15), pad=((18,0),(15,0)))],
                         [sg.Slider(range=(1, 100), orientation='v', size=(16, 25), pad=((10, 0),(10,25)), default_value=0, key='T',enable_events=True),
                          sg.Slider(range=(1, 100), orientation='v', size=(16, 25), pad=((10,5),(10,25)), default_value=0, key='H',enable_events=True)]
                         ]

    left_width_layout = [[
        sg.Text('Left', font=('',15), pad=((30,0),(8,0))), sg.Slider(range=(1, 100), orientation='h', size=(46, 20),pad=((10,26),10),
                                                                 default_value=0, key='L',enable_events=True)],
        [sg.Text('Width', font=('',15), pad=((30,0),(0,0))), sg.Slider(range=(1, 100), orientation='h', size=(46, 20), pad=((10,26),(0,10)),
                                                                   default_value=100, key='W',enable_events=True)]]

    cam_flip_layout = [sg.Text('Camera', font=('',18), pad=(5,(16,27))),
                       sg.Spin(list(range(0,len(valid_ids))), initial_value=0, font=('',20), pad=((0,30),(18,27)), key='ID',
                               background_color='#303050', text_color='#FFFFFF', enable_events=True),
                       sg.Checkbox(default=True, text='H-Flip',  font=('',18), pad=((0,30),(18,27)), key='hflip',enable_events=True)]

    chroma_layout = [
        [sg.Text('Hue LoPass', font=('',15), pad=(10,(12,0))),
         sg.Slider(range=(1, 255),orientation='h',size=(20, 20), default_value=40,key='hue_loPass',enable_events=True)],
        [sg.Text('Hue HiPass', font=('',15), pad=(10,(12,0))),
         sg.Slider(range=(1, 255), orientation='h',size=(20, 20), default_value=90, key='hue_hiPass',enable_events=True)],
        [sg.Text('Sat LoPass', font=('',15), pad=(10,(12,0))),
         sg.Slider(range=(1, 255), orientation='h',size=(20, 20), default_value=50, key='sat_loPass',enable_events=True)],
        [sg.Text('Bright LoPass', font=('',15), pad=(10,(12,0))),
         sg.Slider(range=(1, 255), orientation='h',size=(20, 20), default_value=50, key='bright_loPass',enable_events=True)],
        [sg.Text('Rotate', font=('',15), pad=(10,(12,0))),
         sg.Slider(range=(0, 3),orientation='h',size=(20, 20), default_value=0, key='rotate', enable_events=True)],
        [sg.Text('Scale', font=('',15), pad=(10,(12,0))),
         sg.Slider(range=(1, 4),orientation='h',size=(20, 20), default_value=2, resolution = 0.25, key='scale', enable_events=True)],
    ]

    chroma_camera_layout = [[sg.Column(chroma_layout, element_justification='right', pad=(20,15))],
                            cam_flip_layout]



    layout= [[sg.Column(left_width_layout, element_justification='right')],
             [sg.Column(top_height_layout, element_justification='center'),
              sg.Column(chroma_camera_layout, element_justification='center')]
             ]

    return layout