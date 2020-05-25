
# coding: utf-8

# In[1]:


# Statistical Leaning
# Image Recognition
# By: Farhad Salemi



# In[2]:


#Boilerplate
import pandas as pd
import numpy as np

# scikit-learn functions
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, recall_score, precision_score

# TensorFlow / Keras functions
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10

#Load CIFAR 10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print('X shapes: ', X_train.shape, X_test.shape)
print('y shapes: ', y_train.shape, y_test.shape)


# In[3]:


#Import ConfigProto
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[4]:


#Reduce CIFAR 10 to Frogs and Ships
frog = 6
ship = 8
train_ind = np.where((y_train == frog) | (y_train == ship))[0]
test_ind = np.where((y_test == frog) | (y_test == ship))[0]
y_train = y_train[train_ind]
X_train = X_train[train_ind]
y_test = y_test[test_ind]
X_test = X_test[test_ind]

#Relabel frog as 0 and ship as 1 for binary classification
y_train[y_train == frog] = 0
y_train[y_train == ship] = 1
y_test[y_test == frog] = 0
y_test[y_test == ship] = 1

#Create Test, Training, and Validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=.5, stratify=y_train, random_state=1)

#Reshaping input data
X_train = X_train.reshape(X_train.shape[0], -1)
X_valid = X_valid.reshape(X_valid.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#Reshape response data
y_train = y_train.ravel()

#Standardize input data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
 
#Reduce input data dimensions with PCA
pca = PCA(n_components=100, random_state=1)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
X_valid = pca.transform(X_valid)
print('X shapes: ', X_train.shape, X_valid.shape, X_test.shape)


# In[5]:


#Initial KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

preds = knn.predict(X_test)
print('KNN Accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
print('KNN Recall: {:.3f}'.format(recall_score(y_test, preds)))
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
print('KNN F1: {:.3f}'.format(2 * (precision * recall) / (precision + recall)))


# In[6]:


#Optimized KNN
knn = KNeighborsClassifier()
ks = np.arange(3, 11)
gs = GridSearchCV(knn, param_grid={'n_neighbors':ks}, scoring='accuracy', cv = 5)
gs.fit(X_train, y_train)
 
print(gs.best_params_)

preds = gs.predict(X_test)
print('KNN Accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
print('KNN Recall: {:.3f}'.format(recall_score(y_test, preds)))
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
print('KNN F1: {:.3f}'.format(2 * (precision * recall) / (precision + recall)))


# In[7]:


#Initial Random Forest
rf = RandomForestClassifier(n_estimators  = 100)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
print('Random Forest Recall: {:.3f}'.format(recall_score(y_test, preds)))
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
print('Random Forest F1: {:.3f}'.format(2 * (precision * recall) / (precision + recall)))


# In[8]:


#Optimized Random Forest
rf = RandomForestClassifier(max_features='sqrt')
trees = [200,300]
depths = [9,10,11]

gs = GridSearchCV(rf,param_grid={'n_estimators':trees,'max_depth':depths},scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)
gs.best_params_


# In[9]:


rf = RandomForestClassifier(n_estimators=300, max_depth=10)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
print('Random Forest Recall: {:.3f}'.format(recall_score(y_test, preds)))
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
print('Random Forest F1: {:.3f}'.format(2 * (precision * recall) / (precision + recall)))


# In[10]:


#Initial Gradient Boosted
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
 
preds = gb.predict(X_test)
print('Gradient Boosting Accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
print('Gradient Boosting Recall: {:.3f}'.format(recall_score(y_test, preds)))
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
print('Gradient Boosting F1: {:.3f}'.format(2 * (precision * recall) / (precision + recall)))


# In[ ]:


#Optimized Gradient Boosted
gb = GradientBoostingClassifier()
trees = [200, 300]
learning = [.05, .1, .2]
gs = GridSearchCV(gb, param_grid={'n_estimators':trees, 'learning_rate':learning},
scoring='accuracy', cv = 5)
gs.fit(X_train, y_train)

gs.best_params_
 
preds = gs.predict(X_test)
print('Gradient Boosting Accuracy: {:.3f}'.format(accuracy_score(y_test, preds)))
print('Gradient Boosting Recall: {:.3f}'.format(recall_score(y_test, preds)))
precision = precision_score(y_test, preds)
recall = recall_score(y_test, preds)
print('Gradient Boosting F1: {:.3f}'.format(2 * (precision * recall) / (precision + recall)))


# In[ ]:


#Changing the shape of y_train for Neural Net
y_train = y_train.reshape(-1, 1)


# In[ ]:


#Initial Neural Net
model2 = Sequential()
model2.add(Dense(50, activation='relu', input_shape=(X_train.shape[1], )))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(10, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['mean_squared_error', 'accuracy'])

model2.fit(X_train, y_train, epochs=20, batch_size=50,
           validation_data=(X_valid, y_valid), verbose = 2)

model2.evaluate(X_test, y_test, verbose = 2)


# In[ ]:


history = model2.history.history
plt.title('Model Accuracy')
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


plt.title('Model MSE')
plt.plot(history['mean_squared_error'], label='Training')
plt.plot(history['val_mean_squared_error'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


#Optimized Neural Net
from tensorflow import keras
model3 = Sequential()
model3.add(Dense(100, activation='relu', input_shape=(X_train.shape[1], ), 
                 kernel_regularizer=keras.regularizers.l1(.001)))
model3.add(Dropout(rate=.5))
model3.add(Dense(25, activation='relu',
                 kernel_regularizer=keras.regularizers.l1(.001)))
model3.add(Dropout(rate=.5))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['mean_squared_error', 'accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model3.fit(X_train, y_train, epochs=15, batch_size=10,
          validation_data=(X_valid, y_valid),
          callbacks=[early_stopping], verbose = 2)

model3.evaluate(X_test, y_test, verbose = 2)


# In[ ]:


history = model3.history.history
plt.title('Model Accuracy')
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


plt.title('Model MSE')
plt.plot(history['mean_squared_error'], label='Training')
plt.plot(history['val_mean_squared_error'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


plt.title('Model Loss')
plt.plot(history['loss'], label='Training')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


#Changing the shape of y_train for Convolutional Neural Net
#y_train = y_train.reshape(-1, 1)


# In[ ]:


#Preprocessing data for Convolutional Neural Net
X_train = pca.inverse_transform(X_train)
X_test = pca.inverse_transform(X_test)
X_valid = pca.inverse_transform(X_valid)
 
X_train = X_train.reshape(-1, 32, 32, 3)
X_valid = X_valid.reshape(-1, 32, 32, 3)
X_test = X_test.reshape(-1, 32, 32, 3)
print('X shapes: ', X_train.shape, X_valid.shape, X_test.shape)


# In[ ]:


#Initial Convolutional Neural Net
model1 = Sequential()
model1.add(Conv2D(16, kernel_size=3, activation='relu', 
                 padding='same', input_shape=(32,32,3)))

model1.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model1.add(Flatten())
model1.add(Dense(1, activation='sigmoid'))

model1.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['mean_squared_error', 'accuracy'])
model1.fit(X_train, y_train, epochs=20, batch_size=10, 
           validation_data=(X_valid, y_valid), verbose = 2)
model1.evaluate(X_test, y_test, verbose = 2)


# In[ ]:


history = model1.history.history
plt.title('Model Accuracy')
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


plt.title('Model MSE')
plt.plot(history['mean_squared_error'], label='Training')
plt.plot(history['val_mean_squared_error'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


plt.title('Model Loss')
plt.plot(history['loss'], label='Training')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


#Optimized Convolutional Neural Net
model3 = Sequential()
model3.add(Conv2D(64, kernel_size=2, activation='relu', strides=1, 
                 padding='same', input_shape=(32,32,3),
                 kernel_regularizer=keras.regularizers.l2(.001)))
model3.add(Dropout(rate=.5))
model3.add(MaxPooling2D(pool_size=(4, 4), strides=2))

model3.add(Conv2D(128, kernel_size=4, activation='relu', 
                 padding='same',
                 kernel_regularizer=keras.regularizers.l2(.001)))
model3.add(Dropout(rate=.5))
model3.add(MaxPooling2D(pool_size=(2, 2), strides=4))

model3.add(Conv2D(64, kernel_size=4, activation='relu', 
                 padding='same',
                 kernel_regularizer=keras.regularizers.l2(.001)))
model3.add(Dropout(rate=.5))
model3.add(MaxPooling2D(pool_size=(2, 2), strides=4))

model3.add(Flatten())
model3.add(Dense(1, activation='sigmoid'))

model3.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['mean_squared_error', 'accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model3.fit(X_train, y_train, epochs=9, batch_size=50, 
           validation_data=(X_valid, y_valid), verbose = 2, callbacks=[early_stopping])
model3.evaluate(X_test, y_test, verbose = 2)


# In[ ]:


history = model3.history.history
plt.title('Model Accuracy')
plt.plot(history['accuracy'], label='Training')
plt.plot(history['val_accuracy'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


plt.title('Model MSE')
plt.plot(history['mean_squared_error'], label='Training')
plt.plot(history['val_mean_squared_error'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[ ]:


plt.title('Model Loss')
plt.plot(history['loss'], label='Training')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epochs')
plt.legend()
plt.show()

