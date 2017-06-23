import keras
import test
import os,cv2
import numpy as np
x,y = test.training_set()
path = 'data2/x/'
store = 'predicted/'
files = os.listdir(path)
m = keras.models.load_model('sr_deconv_net.h5')
Y = m.predict(x, batch_size=16)
for i in range(99):
    print(Y[i] - y[i])
    print(Y[i].shape)
    cv2.imwrite('predicted/'+files[i], np.uint8(Y[i]))
