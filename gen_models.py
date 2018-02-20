
from keras.models import Sequential
from keras.layers import Dense
import keras
import sys

input_dim = int(sys.argv[1])
model_name = 'my_model' if len(sys.argv) == 2 else sys.argv[2]

NETWORKS = [
    [800, 400, 200, 50, 2],
    [400, 200, 50, 2],
    [200, 50, 2],
    [50, 2],
    [800, 400, 200, 50, 1],
    [400, 200, 50, 1],
    [200, 50, 1],
    [50, 1]
]

for NETWORK in NETWORKS:
    model = Sequential()
    for i in range(len(NETWORK)):
        if i == 0:
            model.add(Dense(NETWORK[i], input_dim=input_dim, activation='relu'))
        else :
            model.add(Dense(NETWORK[i], activation='relu' if i < len(NETWORK) - 1 else 'sigmoid'))
    model.compile(loss=keras.losses.categorical_crossentropy if NETWORK[-1] == 2 else keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    model.save('{0}_{1}.h5'.format(model_name, '_'.join([str(n) for n in NETWORK])))