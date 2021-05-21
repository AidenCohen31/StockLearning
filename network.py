import numpy as np
import math
import matplotlib.pyplot as plt

class NueralNetwork:
    layers = []
    data = []
    answers = []
    def __init__(self,data,answers):
        self.data = data
        self.answers =np.array([[answers[i]] for i in range(len(answers))])

    def createTraditionalNetwork(self,num):
        self.layers.append(InputLayer(self.data,1,2))
        self.layers.append(ActivationLayer())
        for i in range(num):
            self.layers.append(FullyConnectedLayer(num))
            self.layers.append(ActivationLayer())
        self.layers.append(OutputLayer(2,1))
        self.layers.append(ActivationLayer())
    def run(self, limit):
        true_error = float("inf")
        #abs(np.sum(true_error)) > limit
        for i in range(1000):
            for i in range(len(self.layers)):
                output = self.layers[i].forwardp()
                if(i != len(self.layers) - 1):
                    self.layers[i+1].inputs = np.array(output)
                else:
                    vector = np.vectorize(np.sin)
                    errorarr = np.square(output-vector(self.data))
                    true_error = np.sum(errorarr)/len(errorarr)
                    #print(true_error)
                    '''
                    plt.plot(self.data,output, label = "line1")
                    x = np.arange(0., 5., 0.2)
                    plt.plot(x,np.sin(x), label = "line2")
                    plt.show()
                    '''
            dactivate = []
            activate = []
            error = 0
            for i in reversed(range(len(self.layers))):
              
                if(str(type(self.layers[i]))== "ActivationLayer"):
                    activate = np.array(self.layers[i].nodes)
                    dactivate = np.array(self.layers[i].backp())
                elif(str(type(self.layers[i])) == "OutputLayer"):
                    
                    print(np.array([self.answers[0]]))
                    error = self.layers[i].backp(np.array([self.answers[0]]),activate, dactivate)
                else:
                    error = self.layers[i].backp(error, activate ,dactivate)
                    
        
    def getNextDataPoint(self):
            pass

class PrettyType(type):
    def __repr__(self):
        return self.__name__             
class Layer(metaclass=PrettyType):

    #alpha is learning rate
    alpha = .001
    delta = 1
    def init(self):
        inputs = []
        nodes = []
        pass
    def nodes(self):
        return nodes
    def cost(self, data):
        x = lambda residual: .5*(residual)**2 if residual <= self.delta else self.delta * Math.abs(residual) - .5(self.delta**2)
        return x(data)
    def dcost(self, data):
        x = lambda x: x if x <= self.delta else self.delta
        return x(data)
    def relu(self, x):
        return x if x > 0 else self.slope * x
    def drelu(self,x):
        return 1 if x > 0 else self.slope


class FullyConnectedLayer(Layer):
    def __init__(self, number):
        self.inputs = []
        self.bias = 0
        std = math.sqrt(2.0/number)
        self.weights = np.array([np.random.randn(number)*std for i in range(number)])
    def forwardp(self):
        print("FC inputs")
        print(self.inputs)
        print("FC weights")
        print(self.weights)
        print("FC Output")
        print(np.matmul(self.inputs, self.weights) + self.bias)
        return np.matmul(self.inputs, self.weights) + self.bias
    def backp(self,error, activation, dactivation):
        curr_error = error[:,0].reshape(len(self.nodes),1)
        weights = self.weights[0].reshape(2,1)
        input_error = np.multiply(np.matmul(curr_error,weights.transpose()),dactivation)
        delta = input_error[:,0].reshape(len(self.nodes),1)
        for i in range(1, len(input_error[0])):
            delta = delta + input_error[:,i].reshape(len(self.nodes),1)
        for i in range(len(self.weights)):
            for j in range(len(input_error[0])):
                weight_error = np.sum(np.multiply(input_error[:,j].reshape(len(self.nodes),1), activation[:,i].reshape(len(self.nodes),1)) * self.alpha)/float(len(input_error))
                self.weights[i,j] = self.weights[i,j] - weight_error
                self.bias -= weight_error
            curr_error = error[:,i].reshape(len(self.nodes),1)
            weights = self.weights[i].reshape(2,1)
            input_error = np.multiply(np.matmul(curr_error,weights.transpose()),dactivation)
            if(i!= 0):
                temp = input_error[:,i].reshape(len(self.nodes),1)
                for j in range(1, len(input_error[0])):
                    temp = temp + input_error[:,j].reshape(len(self.nodes),1)
                delta = np.concatenate((delta,temp), axis = 1)
        print("delta")
        print(delta)
        return delta

class ActivationLayer(Layer):
    slope = 0
    def __init__(self,slope):
        self.slope = slope
        self.nodes = []
        self.inputs = []
    def __init__(self):
        self.slope = .01
        self.nodes = []
        self.inputs = []
    def is_num(self,i):
        try:
            float(i)
            return True
        except ValueError:
            return False
    def fhelper(self,arr,i,forward):
        temp = []
        if(self.is_num(str(arr[0]))):
            for j in arr:
                temp.append(self.relu(j) if forward else self.drelu(j))
            return temp
        for j in arr:
            self.nodes.append(self.fhelper(j,i,forward))
            i=i+1
        return self.nodes
    def forwardp(self):
        self.nodes = []
        #print("activation")
        self.fhelper(self.inputs,0, True)
        #print(self.nodes)
        return self.nodes
    def backp(self):
        self.nodes = []
        #print("dactivation")
        self.fhelper(self.inputs, 0, False)
        return self.nodes

class InputLayer(Layer):
    def __init__(self, inputs,number,numoutputs):
        self.nodes = np.array([inputs[0]])
        std = math.sqrt(2.0/number)
        self.weights = np.array([np.random.rand(numoutputs) * std]) 
        self.bias = 0
    def forwardp(self):
        '''
        print("input inputs")
        print(self.nodes)
        print("input")
        print(self.nodes)
        print("input weights")
        print(self.weights)
        print("input output")
        print(np.matmul(self.nodes.reshape(len(self.nodes),1), self.weights) + self.bias)
        '''
        return np.matmul(self.nodes.reshape(len(self.nodes),1), self.weights) + self.bias 
    def backp(self,error, activation, dactivation):
        delta = np.multiply(np.matmul(error,self.weights), dactivation)
        '''
        print("delta")
        print(delta)
        print("new weights")
        '''
        self.weights -= self.alpha * np.multiply(delta, activation)
        #print(self.weights)
        #print("new bias")
        self.bias -= self.alpha * np.average(delta)
        #print(self.bias)
        return delta
    

class OutputLayer(Layer):
    def __init__(self, number,numoutputs):
        self.inputs = []
        self.bias = 0
        std = math.sqrt(2.0/number)
        self.weights = np.array([np.random.randn(1)*std for i in range(number)])
    def forwardp(self):
        '''
        print("output inputs")
        print(self.inputs)
        print("output weights")
        print(self.weights)
        print("Output output")
        '''        
        return np.matmul(self.inputs, self.weights) + self.bias
    def backp(self, expected, activation, dactivation):
        vector = np.vectorize(self.dcost)
        answers = np.multiply(vector(np.subtract(self.forwardp(),expected)), dactivation)
        print(answers)
        self.weights = self.weights - (self.alpha * np.multiply(answers, activation))
        self.bias -= self.alpha * answers
        return answers
    

'''
x = np.arange(0., 5., 0.2)
plt.plot(x,np.sin(x))
plt.title('Sine wave')
'''
x = np.random.random_sample(100)*10
y = np.sin(x)
 
#plt.show()

nueral = NueralNetwork(x,y)

nueral.createTraditionalNetwork(0)
nueral.run(.1)

        
    
