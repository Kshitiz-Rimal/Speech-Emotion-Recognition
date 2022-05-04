import tkinter as tk
import tkinter.messagebox
import pyaudio
import wave
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from IPython.display import Audio

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


from playsound import playsound
from keras.models import model_from_json

import extraction_functions as efs
#from main import ExternalAudio




class RecAUD:

    def __init__(self,chunk=3024 , frmat=pyaudio.paInt16, channels=2, rate=44100, py=pyaudio.PyAudio()):

        # Start Tkinter and set Title
        self.main = tkinter.Tk()
        #self.collections = []
        self.main.geometry('500x300')
        self.main.title('Record')
        self.CHUNK = chunk
        self.FORMAT = frmat
        self.CHANNELS = channels
        self.RATE = rate
        self.p = py
        #self.frames = []
        self.st = 1
        self.stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)

        # Set Frames
        self.buttons = tkinter.Frame(self.main, padx=130, pady=20)

        # Pack Frame
        self.buttons.pack(fill=tk.BOTH)



        # Start and Stop buttons
        self.strt_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Start Recording',bg = "black",fg="white", command=lambda: self.start_record())
        self.strt_rec.grid(row=0, column=0, padx=50, pady=5)
        self.stop_rec = tkinter.Button(self.buttons, width=10, padx=10, pady=5, text='Stop Recording',bg = "black",fg="white", command=lambda: self.stop())
        self.stop_rec.grid(row=1, column=0, columnspan=1, padx=50, pady=5)

        tkinter.mainloop()

    def start_record(self):
        self.st = 1
        self.frames = []
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        while self.st == 1:
            data = stream.read(self.CHUNK)
            self.frames.append(data)
            print("* recording")
            self.main.update()

        stream.close()

        wf = wave.open('test_recording.wav', 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def stop(self):
        self.st = 0


# Create an object of the ProgramGUI class to begin the program.
disp = RecAUD()


# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


#getting the features for recorded audio
feature = efs.get_features('test_recording.wav')
X= np.array([])
for ftr in feature:
   X=np.hstack((X,ftr))
print(X)


# X = X.reshape(-1,1)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# #shaping for model
X = X.reshape(1,-1)
print(X)


#this block is for inverse encoding of the final output labels
data = pd.read_csv('features.csv')
Y = data['category'].values
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()


audio_predict = loaded_model.predict(X)
print(audio_predict)
final_label = encoder.inverse_transform(audio_predict)


# final_class = ExternalAudio(audio_predict)
# final_label = final_class.predictRecorded()
# print(final_label)


audio_df = pd.DataFrame(columns=['AudioLabel'])
audio_df['AudioLabel'] = final_label.flatten()

#for tess 
# audio_df.AudioLabel.replace({0:'neutral', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 1:'surprise'}, inplace=True)

#for tess removed
audio_df.AudioLabel.replace({0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'surprise'}, inplace=True)
print(audio_df)