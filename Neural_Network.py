import numpy as np
import pandas as pd

def load_data(start,end):
    data=pd.read_csv('digit_data/train.csv')
    data=np.array(data)
    label=data[start:end,0]
    x=data[start:end,1:]
    x=x/255
    y=np.zeros((label.shape[0],10))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if(label[i]==j):
                y[i,j]=1
    return x, y,label

def relu(x):
    return np.where(x>0,x,0)

def der_relu(x):
    return np.where(x>0,1,0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dersig(x):
    return (x*(1-x))

def softmax(input,w,b):
    d=np.exp(np.dot(input,w)+b)
    for i in range(d.shape[0]):
        dem=np.sum(d[i])
        d[i]=(d[i]/dem)
    return d 

def dense(input,w,b,function):
    if(function=="relu"):
        return relu(np.dot(input,w)+b)
    elif(function=="sigmoid"):
        return sigmoid(np.dot(input,w)+b)
    elif(function=="softmax"):
        return softmax(input,w,b)

def der(x,function):
    if(function=="relu"):
        return der_relu(x)
    elif(function=="sigmoid"):
        return dersig(x)
    
def sig_loss(a,y):
    m,n=a.shape
    cost=0
    for i in range(n):
        cost+=(-(y[i]*np.log(a[0,i]))-((1-y[i])*np.log(1-a[0,i])))
    return cost/n

def soft_loss(a,y):
        m,n=a.shape
        d=(-1*(np.log(np.dot(((a.T)*y),np.ones((m,1))))))
        return np.sum(d)/n

def loss(a,y,loss_name):
    if(loss_name=="soft_loss"):
        return soft_loss(a,y)
    elif(loss_name=="sig_loss"):
        return sig_loss(a,y)
    
class nn:
    def __init__(self,*args):

        self.args=args
        self.nb_layers=(len(self.args)-1)
        self.random_range=self.args[0][1]

        self.weight=[]
        self.bias=[]
        self.activation=[]
        self.function_of_activation=[]

        for i in range(self.nb_layers):
            self.weight.append(np.random.uniform(-self.random_range,self.random_range,(self.args[i][0],self.args[i+1][0])))
            self.bias.append(np.random.uniform(-self.random_range,self.random_range,self.args[i+1][0]))
            self.activation.append(np.zeros((self.args[i+1][0])))
            self.function_of_activation.append(self.args[i+1][1])       

    def forw(self,inputs):
        for i in range(self.nb_layers):
            if(i==0):
                self.activation[i]=dense(inputs,self.weight[i],self.bias[i].T,self.function_of_activation[i]).T
            else:
                if(i==(self.nb_layers-1)):
                    self.activation[i]=dense(self.activation[i-1].T,self.weight[i],self.bias[i].T,self.function_of_activation[i]).T
                else:
                    self.activation[i]=dense(self.activation[i-1].T,self.weight[i],self.bias[i].T,self.function_of_activation[i]).T

    
    def backword(self,inputs,targets,alpha):

        m=inputs.shape[0]
        err=[]
        delta=[]
        for i in range(self.nb_layers):
            layer_index=(self.nb_layers-1)-i
            if(i==0):
                err.append(self.activation[layer_index]-targets.T)
                delta.append(np.dot(self.weight[layer_index],err[i]))
            else:
                if(layer_index==0):
                    err.append(delta[i-1]*der(self.activation[layer_index],self.function_of_activation[layer_index]))
                else:
                    err.append(delta[i-1]*der(self.activation[layer_index],self.function_of_activation[layer_index]))
                    delta.append(np.dot(self.weight[layer_index],err[i]))

        for i in range(self.nb_layers):
            layer_index=(self.nb_layers-1)-i
            if(layer_index==0):
                self.weight[layer_index]-=alpha*(np.dot(inputs.T,err[i].T)/m)
            else:
                self.weight[layer_index]-=alpha*(np.dot(self.activation[layer_index-1],err[i].T)/m)
            self.bias[layer_index]-=alpha*(np.sum(err[i],axis=1)/m)

    def train(self,inputs,target,alpha,n,cost):
        for i in range(n):
            self.forw(inputs)
            self.backword(inputs,target,alpha)
            print(f"itaration {i+1} loss={loss(self.activation[self.nb_layers-1],target,cost)}")

    def predict(self, inputs):
        self.forw(inputs)
        return self.activation[self.nb_layers-1]

input_layer = [784,0.5]
hidden_layer1 = [25,"relu"]
hidden_layer2=[15,"relu"]
output_layer = [10,"softmax"]
cost="soft_loss"

network = nn(input_layer, hidden_layer1, hidden_layer2 ,output_layer)

inputs,targets,labels=load_data(0,40000)

learning_rate =0.2
epochs = 300

network.train(inputs, targets, learning_rate, epochs,cost)

prex,prey,prelabel=load_data(40000,42000)

predicted=network.predict(prex)
predicted=predicted.T

count=0

for i in range(predicted.shape[0]):
    for j in range(10):
        if(predicted[i][j]==np.max(predicted[i])):
            if(prelabel[i]!=j):
                count+=1
            print(f"label {prelabel[i]} , predicted {j}")

print(f"accurcy= {(1-(count/predicted.shape[0]))*100}%")