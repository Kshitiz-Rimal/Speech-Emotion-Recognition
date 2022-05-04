import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa.display
from IPython.display import Audio
import warnings

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

import extraction_functions as efs
import RNNLstm2 as lstm2

warnings.filterwarnings('ignore')



#Defining the paths of the dataset used
Ravdess="C:\\Users\Dell\\Desktop\\major project\\SERsystem\\audio_speech_actors_01-24\\"
Tess="C:\\Users\\Dell\\Desktop\\major project\\SERsystem\\TESS\\"
Savee="C:\\Users\\Dell\\Desktop\\major project\\SERsystem\\SAVEE\\"



ravdess_directory_list = os.listdir(Ravdess)
file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
    # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2])-1)
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
#Ravdess_df.Emotions.replace({0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 7:'surprise'}, inplace=True)
#Ravdess_df.to_csv("Ravdess_file.csv",index=False)
Ravdess_df.head()



#loading the dataset in vs code
paths = []
labels = []
for dirname, _, filenames in os.walk(Tess):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        if label=='ps':
            labels.append('surprise')
        else:
            labels.append(label.lower())
    if len(paths) == 2800:
        break
# dataframe for emotion of files
emotion_df = pd.DataFrame(labels, columns=['Emotions'])
# dataframe for path of files.
path_df = pd.DataFrame(paths, columns=['Path'])
Tess_df = pd.concat([emotion_df, path_df], axis=1)
Tess_df.Emotions.replace({'neutral':0, 'happy':2, 'sad':3, 'angry':4, 'fear':5,'disgust':6, 'surprise':7}, inplace=True)
Tess_df.head()


savee_directory_list = os.listdir(Savee)
file_emotion = []
file_path = []

for file in savee_directory_list:
    file_path.append(Savee + file)
    part = file.split('_')[1]
    ele = part[:-6]
    if ele=='a':
        file_emotion.append('angry')
    elif ele=='d':
        file_emotion.append('disgust')
    elif ele=='f':
        file_emotion.append('fear')
    elif ele=='h':
        file_emotion.append('happy')
    elif ele=='n':
        file_emotion.append('neutral')
    elif ele=='sa':
        file_emotion.append('sad')
    else:
        file_emotion.append('surprise')
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Savee_df = pd.concat([emotion_df, path_df], axis=1)
Savee_df.Emotions.replace({'neutral':0, 'happy':2, 'sad':3, 'angry':4, 'fear':5,'disgust':6, 'surprise':7}, inplace=True)
Savee_df.head()



# creating Dataframe using all the 4 dataframes we created so far.
df= pd.concat([Tess_df, Ravdess_df, Savee_df], axis = 0)
df.to_csv("data_path.csv",index=False)
df.head()


#printing the counts of each classes present
print(df['Emotions'].value_counts())



#iterating over the dataset and getting the features in array/table
Z=np.empty((0,90))
Y=[]
i=0
for path, emotion in zip(df.Path, df.Emotions):
  feature = efs.get_features(path)
  #print(feature)
  X= np.array([])
  for ftr in feature:
    X=np.hstack((X,ftr))
  Y.append(emotion)
  Z=np.vstack((Z,X))
  if i%500==0:
        print(i)
  i=i+1



#the extracted data is shown in a table
Features = pd.DataFrame(Z)
Features['category'] = Y
Features.to_csv('features.csv', index=False)
Features.head()



#load X an Y from the dataframe
X = Features.iloc[: ,:-1].values
Y = Features['category'].values


#feature selection,stop the search when only the last 40 feature are left
# lr = LinearRegression(normalize=True)
# lr.fit(X,Y)
# rfe = RFE(estimator = lr, n_features_to_select=60, step=1)
# X = rfe.fit_transform(X,Y)


# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
print(Y)


# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# print(x_train)
# scaling our data with sklearn's Standard scaler
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)
# x_train.shape, y_train.shape, x_test.shape, y_test.shape


# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


#building a model
input_shape = (90,1)
model = lstm2.build_model(input_shape)


history=model.fit(x_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))


#printing the accuracy of the model and the loss and accuracy graph
print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")
epochs = [i for i in range(30)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(20,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()



# As this is a multiclass classification problem onehotencoding our Y.
# encoder = OneHotEncoder()
# Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

# predicting on test data.
pred_test = model.predict(x_test)
y_pred = encoder.inverse_transform(pred_test)
y_test = encoder.inverse_transform(y_test)




# #creating the table for the prediction on test data
Predicted_df = pd.DataFrame(columns=['PredictedLabels', 'ActualLabels'])
Predicted_df['PredictedLabels'] = y_pred.flatten()
Predicted_df['ActualLabels'] = y_test.flatten()
Predicted_df.PredictedLabels.replace({0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 7:'surprise'}, inplace=True)
Predicted_df.ActualLabels.replace({0:'neutral', 1:'calm', 2:'happy', 3:'sad', 4:'angry', 5:'fear', 6:'disgust', 7:'surprise'}, inplace=True)


print(Predicted_df.head(10))
print(classification_report(y_test, y_pred))


#saving the model in json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

