
# coding: utf-8

# In[43]:

import random
import numpy as np
def generateData(rank, max_number = 100):
    raw = []
    if(rank < 0):
        raise
    for x in range(rank):
        x1 = random.randint(0,max_number)
        x2 = random.randint(0,max_number)
        x3 = random.randint(0,max_number)
        isDivided = (x1 + x2 + x3) % 3 == 0
        x4 = 0 if (isDivided) else 1
        raw.append([x1, x2, x3, x4])
    return np.array(raw)


# In[63]:

all_data = generateData(75)
#print(all_data)


# In[45]:

train_data = all_data[:,0:3]
labels = all_data[:,3:4]


# In[47]:

# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense


# In[53]:

model = Sequential()
model.add(Dense(16, input_dim=3))
model.add(Activation('sigmoid'))

model.add(Dense(1))
model.add(Activation('softmax'))


# In[56]:

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[57]:

score = model.evaluate(train_data, labels, batch_size=128)


# In[69]:

predictions = model.predict(np.array([ 89,  85,  12 ]).reshape(1,3))
print(predictions)


# In[62]:

print(predictions)


# In[ ]:



