import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time

df=pd.read_csv('heart.csv')

chest_pain=pd.get_dummies(df['cp'],prefix='cp',drop_first=True)
df=pd.concat([df,chest_pain],axis=1)
df.drop(['cp'],axis=1,inplace=True)
sp=pd.get_dummies(df['slope'],prefix='slope')
th=pd.get_dummies(df['thal'],prefix='thal')
rest_ecg=pd.get_dummies(df['restecg'],prefix='restecg')
frames=[df,sp,th,rest_ecg]
df=pd.concat(frames,axis=1)
df.drop(['slope','thal','restecg'],axis=1,inplace=True)

X = df.drop(['target'], axis = 1)
y = df.target.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import warnings
from sklearn.metrics import accuracy_score

conclusion = []

#adam = optimizers.adam(lr=0.003)
optimizers_list = [[optimizers.Adagrad,0.01,0.02,0.03], [optimizers.SGD,0.01,0.02,0.03], [optimizers.Adam,0.001,0.002,0.003]]

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
		#print ('waktu ',i,j, time.time() - start)
		
		e=[str(optimizers_list[i][0]),optimizers_list[i][j]]
		a.append(e)
		a.append(time.time() - start)

		y_pred = classifier.predict(X_test)
		
		ac=accuracy_score(y_test, y_pred.round())
		#print(i,j,'accuracy of the model: ',ac)
		
		a.append(ac)
	conclusion.append(a)

print(conclusion)

tabel = pd.DataFrame(data={"col1": conclusion[0], "col2": conclusion[1], "col3": conclusion[2]})
tabel.to_csv("./file.csv", sep=',',index=False)