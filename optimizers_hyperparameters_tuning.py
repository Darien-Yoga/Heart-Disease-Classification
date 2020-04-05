#this code is created by citing to "https://www.kaggle.com/rajeshjnv/heart-disease-classification-neural-network"
#created by : ajisumbaga7@gmail.com
#this code enables compiling the existing model with variations of optimizers and hyperparameters(learning rate)
#and calculate the time is needed for each configuration (optimizer and hyperparameter) to run the training process
#so that we can conclude which configuration is the most ideal, by comparing the accuracy and the time of training
	#Optimizer :
	#1. Adagrad optimizer with lr = 0.01 , 0.02 , 0.03
	#2. SGD optimizer with lr = 0.01 , 0.02, 0.03
	#3. Adam optimizer with lr = 0.001, 0.002, 0.003

#LIBRARY
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time #used to time operation

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import warnings
from sklearn.metrics import accuracy_score

#READ THE META DATA
 #store to pandas dataframe
df=pd.read_csv('heart.csv')

#CREATE DUMMY VARIABLES
 #from 13 features -> 22 features
 #Dummy variables are useful as they enable us to use a single regression equation to represent multiple groups.
 #The dummy variables act like 'switches' that turn various parameters on and off in an equation.

chest_pain = pd.get_dummies(df['cp'],prefix='cp',drop_first=True)
df = pd.concat([df,chest_pain],axis=1)
df.drop(['cp'],axis=1,inplace=True)
sp = pd.get_dummies(df['slope'],prefix='slope')
th = pd.get_dummies(df['thal'],prefix='thal')
rest_ecg=pd.get_dummies(df['restecg'],prefix='restecg')
frames=[df,sp,th,rest_ecg]
df=pd.concat(frames,axis=1)
df.drop(['slope','thal','restecg'],axis=1,inplace=True)

X = df.drop(['target'], axis = 1)
y = df.target.values

#SPLIT THE DATA TRAIN AND TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

conclusion = []

#DEFINE THE CONFIGURATIONS OF OPTIMIZER
optimizers_list = [[optimizers.Adagrad,0.01,0.02,0.03], [optimizers.SGD,0.01,0.02,0.03], [optimizers.Adam,0.001,0.002,0.003]]

#LOOPING FOR EACH OF CONFIGURATION
for i in range (len(optimizers_list)):
	a=[]
	for j in range (1,(len(optimizers_list[i]))):
		classifier = Sequential()

		# Adding the input layer and the first hidden layer
		classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu', input_dim = 22))

		# Adding the second hidden layer
		classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu'))

		# Adding the output layer
		classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
		
		# Compiling the ANN
		x = optimizers_list[i][0](lr=optimizers_list[i][j])
		classifier.compile(optimizer = x, loss = 'binary_crossentropy', metrics = ['accuracy'])

		start = time.time()
		classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 100)
				
		e=[str(optimizers_list[i][0]),optimizers_list[i][j]]
		a.append(e)
		a.append(time.time() - start)

		y_pred = classifier.predict(X_test)
		
		#CALCULATE THE ACCURACY OF THE MODEL
		ac=accuracy_score(y_test, y_pred.round())
				
		a.append(ac)
	conclusion.append(a)

#print(conclusion)

#STORE THE RESULT IN CSV
tabel = pd.DataFrame(data={"col1": conclusion[0], "col2": conclusion[1], "col3": conclusion[2]})
tabel.to_csv("./file.csv", sep=',',index=False)
