import random
import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_derivative(data): # calculate derivative of sigmoid function
    return (data * (1 - data))

def create_network(input, hidden, output):
    network = []
    hidden_layer = {"weights": [[random.random() for i in range(0,input)] for j in range(0,hidden)], "biases": [random.random() for i in range(0,hidden)], 
                    'outputs': [0 for i in range(0,hidden)], 'change':[0 for i in range(0,hidden)]}
    output_layer = {"weights": [[random.random() for i in range(0,hidden)] for j in range(0,output)], "biases": [random.random() for i in range(0,output)], 
                    'outputs': [0 for i in range(0,output)], 'change':[0 for i in range(0,output)]}
    network.append(hidden_layer)
    network.append(output_layer)
    return network

def input_data(network, input):
    layer = network[0]
    results = []
    for i in range(0,len(layer['weights'])):
        result = 0
        #print(len(layer['weights']))
        for j in range(0,len(layer['weights'][i])):
            result += (layer['weights'][i][j] * input[j])
            #print(layer['weights'][i][j],"times",input[j],"=",layer['weights'][i][j] * input[j])
        result += layer['biases'][i]
        #print(layer['biases'][i])
        #print(result)
        result = 1 / (1 + math.exp(-1 * result))
        layer['outputs'][i] = result
        #print(result)
        results.append(result)
    #return results
    # repeat process for output layer
    input = results
    print("Input\n",input)
    results = []
    layer = network[1]
    #print("Layer:")
    #print(layer)
    #print("\n\n")
    for i in range(0,len(layer['weights'])):
        print(i)
        result = 0
        #print(len(layer['weights']))
        for j in range(0,len(layer['weights'][i])):
            result += (layer['weights'][i][j] * input[j])
            print(layer['weights'][i][j],"times",input[j],"=",layer['weights'][i][j] * input[j])
        result += layer['biases'][i]
        print("+",layer['biases'][i])
        print(result)
        result = 1 / (1 + math.exp((-1 * result)))
        layer['outputs'][i] = result
        print(result)
        results.append(result)
        print(results)
    return results

def backpropagation(network, expected): # done with one expectation (expectation is number only) at a time so network can learn example by example
    layer = network[1]
    errors = []
    expectations = [0 for j in range(0,len(layer['outputs']))] # use correct class output to create list where each entry is correct output for each output node
    for i in range(0,len(expectations)):
        if(i == expected):
            expectations[i] = 1
    print(expectations)
    for i in range(0,len(layer['outputs'])):
        print("Output",layer['outputs'][i])
        print("Expectation",expectations[i])
        errors.append(layer['outputs'][i] - expectations[i])
    #print("Errors for node",i,"in outer layer\n")
    print(errors)
    print(network)
    print("First part")
    for i in range(0,len(layer['outputs'])):
        print(errors[i])
        print(calculate_derivative(layer['outputs'][i]))
        print(errors[i] * calculate_derivative(layer['outputs'][i]))
        layer['change'][i] = errors[i] * calculate_derivative(layer['outputs'][i])
    print(network)
    layer = network[0]
    errors = []
    #print(network)
    print("Second half")
    for i in range(0,len(layer['weights'])):
        error = 0
        for j in range(0,len(network[1]['weights'])):
            print(network[1]['weights'][j][i])
            print(network[1]['change'][j])
            error += (network[1]['weights'][j][i] * network[1]['change'][j]) #i?
        errors.append(error) # errors for ith neuron in hidden layer
    print("Errors",errors)
    for i in range(0,len(layer['change'])):
        print(layer['outputs'][i])
        print(calculate_derivative(layer['outputs'][i]))
        layer['change'][i] = errors[i] * calculate_derivative(layer['outputs'][i])
    print(network)

def adjust_weights(network, instance, rate):
    layer = network[0]
    inputs = instance[:-1]
    
    for i in range(0,len(layer['weights'])):
        for j in range(0,len(inputs)):
            layer['weights'][i][j] -= rate * layer['change'][i] * inputs[j]
        layer['biases'][i] -= rate * layer['change'][i]

    layer = network[1]
    inputs = [x for x in network[0]['outputs']] #inputs = [neuron['output'] for neuron in network[i - 1]]
    #print(inputs)
    for i in range(0,len(layer['weights'])):
        for j in range(0,len(inputs)):
            layer['weights'][i][j] -= rate * layer['change'][i] * inputs[j]
        layer['biases'][i] -= rate * layer['change'][i]


"""
file = input("Enter file name for Q2 training data: ")
#file = "data.txt"

with open(file) as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

attributes = lines[0].split(",")
instances = [[float(data) for data in  line.split(",")] for line in lines[1:]]
X_train = [x[0] for x in instances]
Y_train = [y[0] for y in instances]
#print(attributes)
#print(instances)
#print(type(instances[0][0]))
numExamples = len(attributes)
theta = [0.0] * numExamples
step = 0.05 
temp = theta

iter = int(len(X_train)/20) 

for i in range(0,iter): # how many batches?
    random.shuffle(instances)
    my_instances = instances[:20]
    sum = 0.0
    for j in range(0,len(my_instances)):
        sum += ((theta[0] + (theta[1] * my_instances[j][0])) - my_instances[j][0])
    temp[0] = theta[0] - (step * (1 / numExamples) * sum)
    sum = 0.0
    for j in range(0,len(my_instances)):
        sum += (((theta[0] + (theta[1] * my_instances[j][0])) - my_instances[j][0]) * my_instances[j][0])
    temp[1] = theta[1] - (step * (1 / numExamples) * sum)
    theta[0] = temp[0]
    theta[1] = temp[1]
    print(theta)

file = input("Enter file name for Q1 test data: ")
#file = "data.txt"

with open(file) as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]
instances = [[float(data) for data in  line.split(", ")] for line in lines[1:]] 

X_test = [x[0] for x in instances]
Y_test = [theta[0] + theta[1]*y[0] for y in instances]
print(X_test)

x = np.linspace(0,max(Y_test),100)
y = theta[0] + (theta[1]*x)
plt.scatter( X_train, Y_train, color = 'blue' )
plt.plot(x, y, '-r', label='Prediction')
plt.scatter( X_test, Y_test, color = 'orange' )
plt.title( 'Linear Regression' )
plt.xlabel( 'Input Variable' )
plt.ylabel( 'Output Variable' )
plt.savefig("Q1_plot.png")
plt.show()
print(theta)
"""
# insert section to create input from txt file
random.seed(1)
"""
input = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]
"""

file = input("Enter file name for Q2 training data: ")
#file = "data.txt"

with open(file) as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

attributes = lines[0].split(",")

try:
    instances = [[print(data) for data in  line.split(",")] for line in lines[1:]]
except TypeError as e:
    pass
instances = [[] for line in lines[1:]]
for i in range(1,len(lines)):
    for j in range(0,len(lines[i].split(","))):
        print(lines[i].split(",")[j])
        data = lines[i].split(",")[j]
        try:
            data = float(data) 
            instances[i].append(data)
        except Exception as e:
            pass

instances = instances[1:]
print(instances)
file = input("Enter file name for Q2 test data: ")
#file = "data.txt"

with open(file) as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

attributes = lines[0].split(",")
test = [[float(data) for data in  line.split(",")] for line in lines[1:]]
print(test)
learn_rate = .15
attributes = len(instances[0]) - 1
#print(attributes)
expected = [i[-1] for i in instances]
#print(expected)
network = create_network(2,4,2)
print(network)
input = instances
# print(input[0][:2]) correct
iter = int(len(instances)/20)
for _ in range(iter):
    random.shuffle(input)
    my_input = input[:20]
    for i in range(0,len(my_input)):
        input_data(network, my_input[i][:attributes])
        #print(network)
        backpropagation(network,my_input[i][-1])
        #print(network)
        adjust_weights(network, my_input[0], learn_rate) 
f = open("Q2_preds.txt", "w")
#file = input("Enter file name for Q1 test data: ")

#test = [[float(data) for data in  line.split(", ")] for line in lines]
for instance in test:
    results = input_data(network, instance[:attributes])
    #print(results)
    class_ = max(results)
    max_index = results.index(class_)
    #print(max_index)
    f.write(str(max_index) + "\n")
f.close()
