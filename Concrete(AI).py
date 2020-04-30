import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks.callbacks import ModelCheckpoint

# PreProcess The DataSet :

Data = pd.read_excel(r'/Users/nikhil/Desktop/Project/Concrete (ANN)/Utilities/Concrete_Data.xls')

Data = Data.rename(columns={'Cement (component 1)(kg in a m^3 mixture)'               : 'Cement',
                              'Blast Furnace Slag (component 2)(kg in a m^3 mixture)' : 'Slag',
                              'Fly Ash (component 3)(kg in a m^3 mixture)'            : 'Fly Ash',
                              'Water  (component 4)(kg in a m^3 mixture)'             : 'Water',
                              'Superplasticizer (component 5)(kg in a m^3 mixture)'   : 'Superplasticizer',
                              'Coarse Aggregate  (component 6)(kg in a m^3 mixture)'  : 'Coarse Aggregate',
                              'Fine Aggregate (component 7)(kg in a m^3 mixture)'     : 'Fine Aggregate',
                              'Age (day)'                                             : 'Age',
                              'Concrete compressive strength(MPa, megapascals)'       : 'Concrete Strength'
                            })

Train_Data = Data[0:926]                                                             # Training Data
Test_Data = Data[926:1030]                                                           # Test Data

Train_Data_Samples  = np.array(Train_Data.iloc[0:, 0:8])                             # Convert the DataFrame Into NumPy Arrays (Only Features)
Train_Data_Strength = np.array(Train_Data.iloc[0:, Train_Data.shape[1]-1])           # Convert the DataFrame Into Numpy Arrays (Only Strength)

Test_Data_Samples   = np.array(Test_Data.iloc[0:, 0:Test_Data.shape[1]-1])
Test_Data_Strength  = np.array(Test_Data.iloc[0:, Test_Data.shape[1]-1])

Scaler = MinMaxScaler(feature_range=(0, 1))                                         # Scaling object with specified range
Scaled_Train_Data = Scaler.fit_transform(Train_Data_Samples)                        # Scaled Train_Data

class PlotLosses (keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0
        self.epoch_iter =[]
        self.training_loss = []
        self.validation_loss = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.epoch_iter.append(self.i)
        self.training_loss.append(logs.get('loss'))
        self.validation_loss.append(logs.get('val_loss'))
        self.i += 1

        plt.plot(self.epoch_iter, self.training_loss, label='Train_Loss', color='deepskyblue')
        plt.plot(self.epoch_iter, self.validation_loss, label='Valid_Loss',color = 'orange')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.draw()
        plt.pause(0.01)


plot_losses = PlotLosses()

## Creating The Model :

Model = Sequential([
    Dense(25, input_shape=(8,), activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])

Model.summary()                                                                                                                                                           # Summary Of Created Framework

MCP = ModelCheckpoint('/Users/nikhil/Desktop/Project/Concrete/Utilities/Model_Weights_lr_2.h5', 'val_loss', 0, True, True, mode='auto', period=1)                                  # Model Checkpoint

Model.load_weights('/Users/nikhil/Desktop/Project/Concrete (ANN)/Utilities/Model_Weights_lr.h5')                                                                                        # Loading Weights

Model.compile(Adam(lr = 0.001), loss='mean_absolute_percentage_error')                                                                                                   # Specifying Optimizer, Cost Function
History = Model.fit(Train_Data_Samples, Train_Data_Strength, validation_split=0.11, batch_size=32, epochs=000, shuffle=True, verbose=2, callbacks=[plot_losses, MCP])   # Training the Model

#Model.save_weights('/Users/nikhil/Desktop/Project/Concrete/Utilities/Model_Weights_lr_3.h5')

Predictions = Model.predict(Test_Data_Samples)
print (r2_score(Test_Data_Strength,Predictions))

plt.show()
plt.scatter(Test_Data_Strength, Predictions, marker='D')
plt.plot([0, 50, 100],[0, 50, 100], color='black')
plt.show()