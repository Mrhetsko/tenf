import os
from os import startfile
import shutil

os.system("winget install --id Git.Git -e --source winget")
os.system("cls")
os.system("winget install --id=Python.Python.3.10  -e")
os.system("cls")

# defining global variables
global num_its
global height
global width
global image_name
global image
global prompt
global inf_steps
global combined_arguments
global num_frames

num_frames = 1
num_its = 1
height = 64
width = 64
prompt = "."
inf_steps = 1
combined_arguments = "test"

os.system("pip install pysimplegui")
os.system("cls")

import PySimpleGUI as sg
import os.path
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory
from itertools import repeat
import atexit
import subprocess
import string
import random

os.system("cls")

sg.theme('DarkBlack')

layout = [
    [sg.Text('CLONING REPO', font=('Arial Bold', 24), pad=(20, (10, 0)))],
    [sg.Text('THIS WILL TAKE SOME TIME, PLEASE WAIT', font=('Arial Bold', 24), pad=(20, (10, 0)))],
]

window = sg.Window('DOWNLOADING, PLEASE WAIT (PROGRAM IS NOT FROZEN)', layout, size=(500, 200), resizable=True,
                   finalize=True)

# clone repo then close warning window
os.system("git clone https://huggingface.co/cerspense/zeroscope_v2_576w")
window.close()
os.system("cls")

os.system("git lfs install")
os.system("cls")
os.system("pip install accelerate -U")
os.system("cls")
os.system("pip install --upgrade torch torchvision")
os.system("cls")
os.system("pip install diffusers transformers")
os.system("cls")

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
os.system("cls")


def generate():
    video_frames = eval(combined_arguments)
    video_path = export_to_video(video_frames)
    shutil.copy(video_path, path_to_save)


layout = [
    [sg.Text('Prompt: ', font=('Arial Bold', 24), pad=(20, (10, 0)))],
    [sg.Input('-Enter Prompt Here-', font=('Arial Bold', 10), size=(15, 2), enable_events=True, key='-PROMPT-',
              expand_x=True, pad=(20, (10, 0)))],
    [sg.Text('Iterations: ', font=('Arial Bold', 20), pad=(20, (10, 0)))],
    [sg.Slider(range=(1, 100), resolution=(1), default_value=1, size=(15, 18), expand_x=True, enable_events=True,
               orientation='horizontal', key='-SL_ITS-', pad=(20, (10, 10)))],
    [sg.Text('Inference Steps: ', font=('Arial Bold', 20), pad=(20, (10, 0)))],
    [sg.Slider(range=(1, 250), resolution=(1), default_value=1, size=(15, 18), expand_x=True, enable_events=True,
               orientation='horizontal', key='-SL_STEPS-', pad=(20, (10, 10)))],
    [sg.Text('Height: ', font=('Arial Bold', 20), pad=(20, (10, 0)))],
    [sg.Slider(range=(64, 2048), resolution=(64), default_value=64, size=(15, 18), expand_x=True, enable_events=True,
               orientation='horizontal', key='-SL_HEIGHT-', pad=(20, (10, 10)))],
    [sg.Text('Width: ', font=('Arial Bold', 20), pad=(20, (10, 0)))],
    [sg.Slider(range=(64, 2048), resolution=(64), default_value=64, size=(15, 18), expand_x=True, enable_events=True,
               orientation='horizontal', key='-SL_WIDTH-', pad=(20, (10, 10)))],
    [sg.Text('Frames: ', font=('Arial Bold', 20), pad=(20, (10, 0)))],
    [sg.Slider(range=(1, 1000), resolution=(1), default_value=1, size=(15, 18), expand_x=True, enable_events=True,
               orientation='horizontal', key='-SL_FRAMES-', pad=(20, (10, 10)))],
    [sg.Button('Cancel', size=(8, 1), font=('Arial Bold', 15), pad=(20, (10, 0))),
     sg.Button('Generate', size=(15, 1), font=('Arial Bold', 15), pad=(20, (10, 0)))]
]

window = sg.Window('SimpleVID', layout, size=(1028, 720), resizable=True, finalize=True)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel':
        torch.cuda.empty_cache()
        break
    elif event == '-SL_ITS-':
        num_its = int(values['-SL_ITS-'])
    elif event == '-SL_STEPS-':
        inf_steps = (values['-SL_STEPS-'])
    elif event == '-SL_HEIGHT-':
        height = (values['-SL_HEIGHT-'])
    elif event == '-SL_WIDTH-':
        width = (values['-SL_WIDTH-'])
    elif event == '-SL_FRAMES-':
        num_frames = (values['-SL_FRAMES-'])
    elif event == 'Generate':
        path_to_save = askdirectory(title='Select Folder')
        prompt = (values['-PROMPT-'])
        height = int(height)
        height = str(height)
        width = int(width)
        width = str(width)
        inf_steps = int(inf_steps)
        inf_steps = str(inf_steps)
        num_frames = int(num_frames)
        num_frames = str(num_frames)
        # combining into one statement to properly feed pipe
        combined_arguments = (
                    "pipe(prompt, num_inference_steps=" + inf_steps + ", height=" + height + ", width=" + width + ", num_frames=" + num_frames + ").frames")
        for _ in range(num_its):
            generate()
        height = float(height)
        height = int(height)
        width = float(width)
        width = int(width)
        inf_steps = float(inf_steps)
        inf_steps = int(inf_steps)
        num_frames = float(num_frames)
        num_frames = int(num_frames)

window.close()
