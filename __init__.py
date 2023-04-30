# example dataset
from sklearn.datasets import load_diabetes
# ML linear algoritmh, its use case is to return a function that relates X and y where Y must a number
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
# In normal programming we have y = f(x) and the programmer given an input writes the algoritmh that calculates the output.
# In ML a big input of x and y is given and the function is generated from the ML model.
# A big amount of data is used to "feed" the ML algoritmh, it uses the data to create and train the model that will interact with the final user.

# Load ML diabetes dataset, the Y and X of expression y = f(x)
dataset = load_diabetes()

# This is the dataset schema, how matrix X and vector y are related
schema = dataset['DESCR']

# Unused because i'm splitting the dataset in train and test
# X uppercase convention because it is a numeric matrix
# X = dataset['data']
# y = dataset['target']

# Split the dataset in train and test
X_train, X_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'])

# X matrix is the entrypoint of the ML model, the input of the function
# y = f(x)
# X ==> y

# convetion is to name the class as the algoritmh and the instance as the model
# LinearRegression is the train agoritmh
# model is the model trained by the algoritmh
model = LinearRegression()

# fit is the method that trains the model, extraxts the pattern and the relations between X and y matrix and vector
model.fit(X_train, y_train)

# Prediction output of the test dataset
# Predict applies the generated algoritmh that relates X and y based on training dataset,
# previosuly the model has been trained with X and y,
# now it's quite able to predict the y of the test dataset
p_test = model.predict(X_test)

# It's a good practice to test and compare the predictions
# also with the dataset used to train the model
p_train = model.predict(X_train)

# Compare the predicted output(y) of the input(X) test dataset
# with the real exptected output(y) of the test dataset
# in order to calculate the error/precision of the trained model
mae_test = mean_absolute_error(y_test, p_test)

# Calculate error/precision also on predictions from train dataset
mae_train = mean_absolute_error(y_train, p_train)

print('MAE test: ', mae_test)
print('MAE train: ', mae_train)
