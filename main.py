# Importing the required libraries
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.optimizers import Adam

# Importing the Datasets
df = pd.read_csv('train.csv')
df_p = pd.read_csv('test.csv')

# Swapping the price and price category columns
cols = list(df.columns)
a, b = cols.index('Price'), cols.index('Price Category')
cols[b], cols[a] = cols[a], cols[b]
df = df[cols]

# Dropping the Unnnecessary columns
df = df.drop(columns=['AddressLine2', 'Locality', 'AddressLine1', 'Street', 'Postcode'])
df_p = df_p.drop(columns=['AddressLine2', 'Locality', 'AddressLine1', 'Street', 'Postcode'])

# Combining the test and train datasets
train = pd.DataFrame(df)
test = pd.DataFrame(df_p)
dataset = pd.concat(objs=[train, test], axis=0, sort=False)

# Splitting the merged dataset into dependent and independent variables
X_m = dataset.iloc[:, 2:9].values
Y_m = dataset.iloc[:, 9].values

# Encoding the independent categorical variables 
labelencoder_X = LabelEncoder()
for i in range(0,7):
    X_m[:, i] = labelencoder_X.fit_transform(X_m[:, i])
onehotencoder = OneHotEncoder(categories = 'auto')
X_m = onehotencoder.fit_transform(X_m).toarray()
X_m = X_m[:, 1:]

# Splitting the variables of test and train dataset
X = X_m[0:142981, :]
Y = Y_m[0:142981, ]
X_p = X_m[142981:204258, :]

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(1000, activation='relu', input_shape=(1614,)))

# Adding the second hidden layer
model.add(Dense(900, activation='relu'))

# Adding the third hidden layer
model.add(Dense(800, activation='relu'))

# Adding the fourth hidden layer
model.add(Dense(700, activation='relu'))

# Adding the fifth hidden layer
model.add(Dense(600, activation='relu'))

# Adding the sixth hidden layer
model.add(Dense(500, activation='relu'))

# Adding the seventh hidden layer
model.add(Dense(200, activation='relu'))

# Adding the output layer
model.add(Dense(1, activation='linear'))

# Compiling the ANN 
opt = Adam(lr = 0.001)
model.compile(optimizer=opt, loss='mean_squared_error')

# Fitting the ANN to the trianing data
result = model.fit(X, Y, epochs = 5, validation_split = 0.22)

# Predicting the results for the test data
Y_pred= model.predict(X_p)






