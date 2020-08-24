# Float Cam

## What is this?
Make your Google Meet, Zoom, or WebEx presentations more engaging with easy live repositioning of a webcam output anywhere on the screen. Comes with live green-screen masking, camera switching, cropping, rotating, scaling, and flipping.

Demo/installation video (credits to my 13 yr old for doing this).

[![Float Cam](https://raw.githubusercontent.com/profteachkids/float_cam/master/float_cam1.jpg)](https://www.youtube.com/embed/-1b9dLhkn5E "Float Cam")

### Installation
- Install Python 3.8 or higher - https://www.anaconda.com/products/individual#Downloads
- Download and extract the files from this repository (zip or git clone)
- Windows Start Menu (Win-R) search and open Anaconda prompt
- From that command prompt navigate to the extracted files

> Setup virtual environment, install requirements, and run! <br>
> Don't miss that period at the end, and those are backslashes

```shell
python -m venv .
.\Scripts\activate.bat
pip install -r requirements.txt
python -m float_cam.py
```
> On MacOS, replace (carlosco with your username and adapt the path according to your setup)
```shell
. /Users/carlosco/anaconda3/etc/profile.d/conda.sh
conda activate base
python -m venv .
source ./bin/activate
pip install -r requirements.txt
python -m float_cam.py
```
All features work, but selective transparency is not possible on MacOS.

## Who/Why?
I am a Chemical Engineering professor at the University of Cincinnati and developed this tool for my 
online lectures. My college students love it, but middle and high school kids whom I teach and 
inspire to love STEM, get an absolute kick from this.

Intense, NO FEAR Learning for Kids - you'd be amazed what your kids can accomplish
 with the right online environment - carloscomail@gmail.com
