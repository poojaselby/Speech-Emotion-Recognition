from ast import main
from asyncio.windows_events import NULL
import streamlit as st
import audio_processing as ap
from pydub import AudioSegment
from uuid import uuid4
import os

import pandas as pd
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")

from scipy import stats
import statistics
from statistics import mode

import librosa
import librosa.display

# to play the audio files
from IPython.display import Audio

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


import matplotlib.pyplot as plt
plt.style.use("ggplot")

import keras
from sklearn.preprocessing import StandardScaler


from glob import glob
import os
import streamlit as st
import soundfile as sf
import librosa
import glob

import matplotlib.pyplot as plt
import librosa
from pathlib import Path

import sounddevice as sd
import wavio
from ml import process_audio_model

#manu code
data=NULL;
sampling_rate=NULL;
def save_uploadedfile(uploadedfile):
     with open(os.path.join("audio",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return uploadedfile.name
def read_audio(file):
            with open(file, "rb") as audio_file:
                audio_bytes = audio_file.read()
            return audio_bytes

def record(duration=5, fs=48000):
          sd.default.samplerate = fs
          sd.default.channels = 1
          myrecording = sd.rec(int(duration * fs))
          sd.wait(duration)
          return myrecording

def save_record(path_myrecording, myrecording, fs):
         wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
         return None
#st.balloons()
st.write("We define a SER system as a collection of methodologies that process and classify speech signals to detect emotions embedded in them.")
st.write("This is our attempt to detect underlying emotions in recorded speech by analysing the acoustic features of the audio data of recordings.")
st.image("images/1.jpg")

st.write(ap.hello())
st.sidebar.title("Speech Emotion Recognition System")
#st.video("video.mp4")
def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=5, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None
st.header(" Record your own voice")
filename= st.text_input("Choose a filename")
if st.button(f"Click to record"):
    if filename=="":
        st.warning("Choose a filename")
    else:
        record_state=st.text("Recording..")
        duration=5
        fs=48000
        myrecording= record(duration, fs)
        record_state.text(f"Saving sample as {filename}.wav")
        path_myrecording=f"D:/SEM 3/Speech_Emotion_Recognition/audio/{filename}.wav"
        save_record(path_myrecording, myrecording, fs)
        record_state.text(f"Done Saved sample as {filename}.mp3")
        st.audio(read_audio(path_myrecording))
        process_audio_model(path_myrecording)
        
st.header("Upload your audio")
uploaded_file = st.file_uploader("Upload audio input file",type=['wav'])
if uploaded_file is not None:
        uploaded_file_name=save_uploadedfile(uploaded_file)
        audio_file = open("D:/SEM 3/Speech_Emotion_Recognition//audio/"+uploaded_file_name, "rb")
        st.audio(audio_file.read())
        audiopath="D:/SEM 3/Speech_Emotion_Recognition//audio/"+uploaded_file_name
        process_audio_model(audiopath)