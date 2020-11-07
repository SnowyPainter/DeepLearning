# It makes a dataset for classifying man or woman
# Dataset properties
# X : trainning datas, contains (height, weight, headround, shoulderwidth)
# Y : correct datas, 0 is man, 1 is woman
# X[i]'s whether man or woman information is Y[i] 

import numpy as np
import random
import csv

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.3f}".format(x)})
def random_float(low, high):
    return random.random()*(high-low) + low

class ManWomanDataset:
    
    def __init__(self):
        pass
    def generate(self,len):
        if(len < 0):
            return
        
        x = np.zeros((len, 4))
        y = np.zeros((len, 1))
        for i in range(len):
            xarr = np.array([0,0,0,0.0])
            c = 0
            if i > len/2: #woman
                c = 1
                xarr[0] = np.random.randint(low = 142, high=185)
                xarr[1] = xarr[0]-100 + np.random.randint(low = -10, high=15)
                xarr[2] = np.random.randint(low = 50, high=58)
                xarr[3] = random_float(2.7, 4.7)
            else: #man
                c = 0
                xarr[0] = np.random.randint(low = 167, high=195)
                xarr[1] = xarr[0]-100 + np.random.randint(low = -15, high=10)
                xarr[2] = np.random.randint(low = 54, high=59)
                xarr[3] = random_float(3.8, 5.4)

            x[i] = xarr
            y[i] = c

        self.X = x
        self.Y = y

    def print_all(self):
        getMsg = lambda y: "man" if y==0 else "woman"
        for i,x in enumerate(self.X):
            print("X:{} Y:{}".format(x, getMsg(self.Y[i])))

    def to_file(self, filename):
        with open(filename, 'w', newline='') as csvout:
            writer = csv.writer(csvout, delimiter = ',')
            writer.writerow([ f for f in ["Height", "Weight", "Head Round", "Shoulder Width", "Woman"]])

            for i,x in enumerate(D.X):
                writer.writerow(np.append(x,D.Y[i]))
            csvout.close()

D = ManWomanDataset() 

D.generate(30000)

D.to_file("./man_woman_dataset.csv")

#D.print_all()