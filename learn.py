import numpy as np
import random
import pandas as pd

class TwoLayerMachine:
    def print_wb(self):
        print("W1 : {}".format(self.W1))
        print("W2 : {}".format(self.W2))
        print("B1 : {}".format(self.B1))
        print("B2 : {}".format(self.B2))
    def gradient_1d(self, f,x):
        h = 1e-4
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            tmpx = x[i]
            x[i] = tmpx + h
            fx1 = f(x)
            x[i] = tmpx - h
            fx2 = f(x)
            g[i] = (fx1 - fx2) / (2*h)
            x[i] = tmpx
        return g
    
    def gradient(self,f,x):
        h = 1e-4
        g = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tmpx = x[i][j]
                x[i][j] = tmpx + h
                fx1 = f(x)
                x[i][j] = tmpx - h
                fx2 = f(x)
                g[i][j] = (fx1 - fx2) / (2*h)
                x[i][j] = tmpx
        return g

    def num_gradient(self, x, t):
        loss_W = lambda W: self.xentropy_loss(x,t)
        return self.gradient(loss_W, self.W1),self.gradient(loss_W, self.W2),self.gradient_1d(loss_W, self.B1),self.gradient_1d(loss_W, self.B2)

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
        print("Input Size : {}".format(xsize))
        print("Hidden Layer Size : {}".format(hiddensize))

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

    def train(self, tx, ty, lr=0.01, epoch=100,steps=1000):
        print("Dataset Shape : {}".format(tx.shape))
        print("Train for dataset {} steps, calculate loss by epoch {}".format(steps, epoch))
        print("The Learning Rate is {}".format(lr))

        #loss_W the correct loss
        log_epoch_loss = []

        for i in range(steps):
            batch_mask = np.random.choice(tx.shape[0], epoch)
            x_batch = tx[batch_mask]
            y_batch = ty[batch_mask]
            w1,w2,b1,b2 = self.num_gradient(x_batch, y_batch)
            
            self.W1 -= lr * w1
            self.W2 -= lr * w2
            self.B1 -= lr * b1
            self.B2 -= lr * b2

            if(i%epoch == 0):
                print("{}th Epoch Learning ... ".format(i/epoch))
                log_epoch_loss.append(self.xentropy_loss(x_batch, y_batch))

        return log_epoch_loss

def csv_to_data(file):
    df = pd.read_csv(file, header=0)
    return list(df.columns.values), df.values

def one_hot_incode(t):
    incoded = np.zeros((t.shape[0], 2))
    for i in range(t.shape[0]):
        incoded[i][int(t[i])] = 1
        
    return incoded

features, data = csv_to_data("man_woman_dataset.csv")

feature = features[0:-1]
ylabel = features[-1]

ty = one_hot_incode(data[:,-1])
tx = data[:,:-1]

dl = TwoLayerMachine(tx.shape[1], 100, 2)

losses_step_by_epoch = dl.train(tx, ty)

for i in range(len(losses_step_by_epoch)):
    print("{}th Epoch Cross-Entropy loss : {}".format(i, losses_step_by_epoch[i]))

# Height Weight HeadRound ShoulderWidth
test_case_woman = np.array([166, 54, 56, 3.995])
test_case_man = np.array([174, 66, 57, 4.552])

test_case = test_case_man

genderY = lambda y: "Man" if(y[0] < y[1]) else "Woman"

print("Test Case : {}".format(test_case))

print("Test case result : {}".format(genderY(dl.predict(test_case))))
print("At last, The Loss : {}".format(dl.xentropy_loss(test_case, np.array([1,0]))))