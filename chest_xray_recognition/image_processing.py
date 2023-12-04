import cv2
import os
import numpy as np


def get_labels(dir):
    data = list()
    dirs = os.listdir(dir)
    for dr in dirs:

        if dr == 'xray':
            label = 0
        elif dr == 'non_xray':
            label = 1
        elif dr == 'dog':
            label = 2
        elif dr == 'eye':
            label = 3
        elif dr == 'cataract':
            label = 4
        elif dr == 'banana':
            label = 5
        sub_dr = dir + dr + '/'
        for file in os.listdir(sub_dr):
            rec = list()
            # full_name=dirs+dr+'/'+file
            rec.append(file)
            rec.append(label)
            data.append(rec)
    data = np.array(data)
    return data

# train_dr='./data/train/'
# test_dr='./data/test/'
# train_data=get_labels(train_dr)
# test_data=get_labels(test_dr)
# np.savetxt('xray_recogntion_train.csv',train_data,delimiter=',',fmt='%s')
# np.savetxt('xray_recogntion_test.csv',test_data,delimiter=',',fmt='%s')

# A = np.array([[5, 6, 1], [2, 0, 8], [4, 9, 3]])
# am = np.argmax(A,axis=1)
# print(am)
# print(A.T)