import numpy as np
import random
import pandas as pd

class TwoLayerMachine:
    def print_wb(self):
        print("W1 : {}".format(self.W1))
        print("W2 : {}".format(self.W2))
        print("B1 : {}".format(self.B1))
        print("B2 : {}".format(self.B2))

    def part_diff(self,f, x):
        h = 1e-4
        return (f(x + h) - f(x - h)) / (2 * h)

    def gradient(self,f,x):
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                
                g[i][j] = self.part_diff(f, x[i][j])
                print("s {}".format(g[i][j]))
        print(g)
        return g

    def xentropy_loss(self,x,t):
        xentropy = lambda y,t : -1*np.sum(t * np.log(y+1e-4))
        
        return xentropy(self.predict(x),t)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x+1e-4))
    
    def softmax(self,x):
        expx = np.exp(x)
        sum_expx = np.sum(expx)
        return expx / sum_expx

    def __init__(self, xsize, hiddensize, outsize):
        self.W1 = np.random.randn(xsize, hiddensize)
        self.W2 = np.random.randn(hiddensize, outsize)

        self.B1 = np.zeros(hiddensize)
        self.B2 = np.zeros(outsize)

    def predict(self, x_test):
        a1 = np.dot(x_test,self.W1) + self.B1
        layer1z = self.sigmoid(a1)

        a2 = np.dot(layer1z, self.W2) + self.B2
        layer2z = self.softmax(a2)

        return layer2z # == y

    def train(self, tx, ty, lr=0.01, epoch=100,steps=10000):
        #loss_W the correct loss
        loss_W = lambda W: self.xentropy_loss(tx, ty)
        log_epoch_loss = []
        for i in range(steps):
            self.W1 -= lr * self.gradient(loss_W, self.W1)
            self.W2 -= lr * self.gradient(loss_W, self.W2)
            self.B1 -= lr * self.gradient(loss_W, self.B1)
            self.B2 -= lr * self.gradient(loss_W, self.B2)
            if(i%epoch == 0):
                log_epoch_loss = log_epoch_loss.append(self.xentropy_loss(random.sample(tx, epoch), random.sample(ty, epoch)))

        return log_epoch_loss

def csv_to_data(file):
    df = pd.read_csv(file, header=0)
    return list(df.columns.values), df.values

def one_hot_incode(t, size):
    incoded = np.zeros((t.shape[0], size))
    for i in range(t.shape[0]):
        incoded[i][int(t[i])] = int(t[i])
        
    return incoded

features, data = csv_to_data("man_woman_dataset.csv")

feature = features[0:-1]
ylabel = features[-1]

ty = one_hot_incode(data[:,-1],2)
tx = data[:,:-1]

#InputSize(input's label size same with hiddensize),  hiddenSize , OutputSize
dl = TwoLayerMachine(tx.shape[1], tx.shape[0], 2)

#losses_step_by_epoch = dl.train(tx, ty)

#for i in range(losses_step_by_epoch.size()):
    #print("{} XEntropy loss : {}".format(i, losses_step_by_epoch[i]))

# Height Weight HeadRound ShoulderWidth
test_case_woman = np.array([166, 54, 56, 3.995])
test_case_man = np.array([174, 66, 57, 4.552])

test_case = test_case_man

genderY = lambda y: "Man" if(y[0] < y[1]) else "Woman"

print("Test case result : {}".format(genderY(dl.predict(test_case))))
print("Cross Entropy Loss : {}".format(dl.xentropy_loss(tx,ty)))