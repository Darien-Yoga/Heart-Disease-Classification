# Heart-Disease-Classification
Classifies the given dataset whether is it a heart disease or not based on the given features

![annual-number-of-deaths-by-cause-3-768x542](https://user-images.githubusercontent.com/37804147/78466981-1a356b00-7732-11ea-9aa2-1aeb968d6a53.png)

From this data we can see that cardiovascular disease (Heart Disease) is the most highest death rate in the world in 2017
and the purpose of this Heart Disease Classification is for Classified Patient who had Heart Disease or not having heart disease.
How we classified the patient had Heart Disease or not from dataset that we have got from 
https://www.kaggle.com/ronitf/heart-disease-uci
And the features from this dataset is

![Feature of Heart Disease](https://user-images.githubusercontent.com/37804147/78467134-b4e27980-7733-11ea-8ee8-c83217e79643.PNG)

1. Age : The person's age in years 
2. Sex : The person's sex (1 = male, 0 = female)
3. cp (Chest Paint): The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
4. trestbps (Resting Blood Pressure): The person's resting blood pressure (mm Hg on admission to the hospital)
5. chol (Serum Cholestoral): The person's cholesterol measurement in mg/dl
6. fbs (Fasting Blood Sugar): The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
7. restecg (Resting Electrocardiographic Results): Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
8. thalach (maximum heart rate achieved)
9. exang (exercise induced angina)
10. oldpeak (ST depression induced by exercise relative to rest)
11. slope = the slope of the peak exercise ST segment
12. ca = number of major vessels (0-3) colored by flourosopy
13. thal = thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
14. target = the value of the model (if the value = 0 (no heart disease), and if the value = 1 (has a heart disease))

Artificial Neural Network

Artificial Neural Network - an imitation to the human brain to learn based on the environment and analyzing incomplete information. Each circle in the input layer is an input to a “neuron”, each circle in the hidden layers. Each node from each circle has its own weight. The product of an input and a weight is the strength of the signal. A neuron is activated by a sum product of its input which would be mapped to an Activation Function, which used a sigmoid function. 
![](images/ANN)
![](images/sigmoid)
The processing ability is stored in inter-unit connection strengths, called weights. Input strength depends on the
weight value. Weight value can be positive, negative or zero. Negative weight means that the signal is reduced or
inhibited. Zero weight means that there is no connection between the two neurons. The weights are adjusted to
obtain the required output. There are algorithms to adjust the weights of ANN to get the required output. This
process of adjusting weights is called learning or training.[1]

Adam Optimizer

Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data. Adam is different to classical stochastic gradient descent. Stochastic gradient descent maintains a single learning rate (termed alpha) for all weight updates and the learning rate does not change during training. A learning rate is maintained for each network weight (parameter) and separately adapted as learning unfolds.[2]
![](images/AdamO)

Stochastic Gradient Descent

Stochastic Gradient Descent was firstly used to save computational power for huge dataset. Main difference between the standard gradient descent and SGD is the gradient update, gradient descent process all the dataset before updating. In contrast, SGD updates the gradient everytime an input is processed
