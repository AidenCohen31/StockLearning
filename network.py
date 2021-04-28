import numpy as np
import math
import matplotlib.pyplot as plt

class NueralNetwork:
    layers = []
    data = []
    answers = []
    def __init__(self,data,answers):
        self.data = data
        self.answers = answers
    def createTraditionalNetwork(self,num):
        self.layers.append(InputLayer(self.data,num))
        self.layers.append(ActivationLayer())
        for i in range(num):
            self.layers.append(FullyConnectedLayer(num))
            self.layers.append(ActivationLayer())
        self.layers.append(OutputLayer(num,1))
        self.layers.append(ActivationLayer())
    def run(self, limit):
        error = float("inf")
        while error > limit:
            for i in range(len(self.layers)):
                output = self.layers[i].forwardp()
                if(i != len(self.layers) - 1):
                    self.layers[i+1].inputs = np.array(output)
                else:
                    plt.plot(self.data,output, label = "line1")
                    x = np.arange(0., 5., 0.2)
                    plt.plot(x,np.sin(x), label = "line2")
                    plt.show()
            dactivate = []
            activate = []
            error = []
            self.answers = np.array([[self.answers[i]] for i in range(len(self.answers))])
            for i in reversed(range(len(self.layers))):
              
                if(str(type(self.layers[i]))== "ActivationLayer"):
                    dactivate = self.layers[i].backp()
                    activate = self.layers[i].nodes
                elif(str(type(self.layers[i])) == "OutputLayer"):
                    error = self.layers[i].backp(self.answers,activate)
                else:
                    error = self.layers[i].backp(error, dactivate,activate)
        
    def getNextDataPoint(self):
        pass

class PrettyType(type):
    def __repr__(self):
        return self.__name__             
class Layer(metaclass=PrettyType):

    #alpha is learning rate
    alpha = .9
    def init(self):
        inputs = []
        nodes = []
        pass
    def nodes(self):
        return nodes
    def cost(self, data, delta):
        x = lambda residual: .5*(residual)**2 if residual <= delta else delta * Math.abs(residual) - .5(delta**2)
        return x(data)
    def dcost(self, data,delta):
        x = lambda x: x if x <= delta else delta
        return x(data)
    def relu(self, x):
        return x if x > 0 else self.slope * x
    def drelu(self,x):
        return 1 if x > 0 else self.slope


class FullyConnectedLayer(Layer):
    def __init__(self, number):
        self.inputs = []
        self.nodes = np.array([0.0 for i in range(number)])
        std = math.sqrt(2.0/len(self.nodes))
        self.weights = np.array([np.random.randn(1)*std for i in range(number)]) 
    def forwardp(self):
        return np.add(np.matmul(self.inputs,self.weights), self.nodes)
    def backp(self,error, activation, dactivation):
        delta = np.multiply(np.matmul(error,self.weights.transpose()), dactivation) 
        input_error = np.matmul(delta, activation) * alpha
        weights = np.subtract(weights, input_error)
        biases = np.subtract(nodes, delta*alpha)
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
    def forwardp(self):
        self.nodes = []
        for i in range(len(self.inputs)):
            row = []
            for j in range(len(self.inputs[0])):
                row.append(self.relu(self.inputs[i][j]))
            self.nodes.append(row)
        return np.array(self.nodes)
    def backp(self):
        output = []
        for i in range(len(self.inputs)):
            row = []
            for j in range(len(self.inputs[0])):
                row.append(self.drelu(self.inputs[i][j]))
            output.append(row)
        return output

class InputLayer(Layer):
    def __init__(self, inputs,number):
        self.nodes = np.array([inputs]).transpose()
        std = math.sqrt(2.0/len(inputs))
        self.weights = np.array([np.random.randn(1)*std for i in range(number)]).transpose()
    def forwardp(self):
        return np.matmul(self.nodes, self.weights)
    def backp(self,error, activation, dactivation): 
        delta = np.matmul(error,self.weights.transpose()) * dactivation
        input_error = np.matmul(delta, activation) * alpha
        weights = np.subtract(weights, input_error)
        biases = np.subtract(nodes, delta*alpha)
        return delta
    

class OutputLayer(Layer):
    def __init__(self, number,numoutputs):
        self.inputs = []
        self.nodes = [0.0 for i in range(numoutputs)]
        std = math.sqrt(2.0/len(self.nodes))
        self.weights = np.array([np.random.randn(1)*std for i in range(number)])
    def forwardp(self):
        return np.add(np.matmul(self.inputs,self.weights), self.nodes)
    def backp(self, expected, activation):
        return np.array(list(map(self.dcost,np.subtract(self.forwardp(),expected) * activation,[1 for i in range(len(expected))])))
    

'''
x = np.arange(0., 5., 0.2)
plt.plot(x,np.sin(x))
plt.title('Sine wave')
'''
x = np.random.random_sample(100)*10
y = np.sin(x)
 
#plt.show()

nueral = NueralNetwork(x,y)

nueral.createTraditionalNetwork(2)
nueral.run(.1)

        
    
