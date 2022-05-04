import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import classification_report

from playsound import playsound
from keras.models import model_from_json

import extraction_functions as efs
#from main import ExternalAudio


def labelIdentifier():
    #load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")


    #getting the features for recorded audio
    feature = efs.get_features('recorded.wav')
    X= np.array([])
    for ftr in feature:
      X=np.hstack((X,ftr))


    # X = X.reshape(-1,1)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #shaping for model
    X = X.reshape(1,-1)


    #this block is for inverse encoding of the final output labels
    data = pd.read_csv('features.csv')
    Y = data['category'].values
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()


    audio_predict = loaded_model.predict(X)
    print(audio_predict)
    final_label = encoder.inverse_transform(audio_predict)


    audio_df = pd.DataFrame(columns=['AudioLabel'])
    audio_df['AudioLabel'] = final_label.flatten()
    #this is for main1
    #audio_df.AudioLabel.replace({0:'neutral',1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 7:'surprise'}, inplace=True)
   
    #this is for main
    #audio_df.AudioLabel.replace({0:'neutral', 1:'surprise', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust'}, inplace=True)
   
    #this is for main with tess disgust removed
    audio_df.AudioLabel.replace({0:'neutral', 1:'happy', 2:'sad', 3:'angry', 4:'fear', 5:'surprise'}, inplace=True)
    label_to_send = audio_df.AudioLabel[0]
    print(audio_df.head())
    
    return label_to_send



