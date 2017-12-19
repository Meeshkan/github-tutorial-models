
from keras.models import Sequential
from keras.layers import Dense

NETWORKS = [
    [2400, 1200, 800, 400, 200, 50, 1],
    [1200, 800, 400, 200, 50, 1],
    [800, 400, 200, 50, 1],
    [400, 200, 50, 1],
    [200, 50, 1],
    [50, 1]
]

for NETWORK in NETWORKS:
    model = Sequential()
    for i in range(len(NETWORK)):
        if i == 0:
            model.add(Dense(NETWORK[i], input_dim=1200, activation='relu'))
        else :
            model.add(Dense(NETWORK[i], activation='relu' if i < len(NETWORK) - 1 else 'sigmoid'))
    model.save('my_model_{0}.h5'.format('_'.join([str(n) for n in NETWORK])))