# Or Nasri 316582246  Niv Nahman 318012564
import numpy as np

INPUT_SIZE = 16
HIDDEN_LAYER_SIZE = 8
OUTPUT_SIZE = 1

def runnet(weights_file, data_file, output_file):
    # Load the weights from wnet.txt
    with open(weights_file, 'r') as file:
        weights = np.fromstring(file.read()[1:-1], sep=' ')

    # Define the activation function (sigmoid)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Define the feedforward function
    def feedforward(input):
        hidden = 0
        for i in range(len(input)):
            hidden += int(input[i]) * weights[i]  
        hidden_activation = sigmoid(hidden)
        output = 0
        input_to_hidden = INPUT_SIZE * HIDDEN_LAYER_SIZE
        hidden_to_output = input_to_hidden + (HIDDEN_LAYER_SIZE * OUTPUT_SIZE)
        for i in range(input_to_hidden, hidden_to_output):
            output += hidden_activation * weights[i] 
        output_weights = weights[input_to_hidden:hidden_to_output]
        threshold = (max(output_weights) + min(output_weights)) / 2
        if output >= threshold:
            return 1
        return 0
    
    # Load the test inputs from testnet0.txt
    test_inputs = np.loadtxt(data_file, dtype=str)

    # Process the test inputs
    results = []
    for input_data in test_inputs:
        processed_input = np.array([int(char) for char in input_data])
        result = feedforward(processed_input)
        results.append(result)
    
    temp = []
    # Print the results
    for result in results:
        if result > 0.5:
            temp.append('1')
        else:
            temp.append('0')

    # Open the file in write mode
    with open(output_file, "w") as file:
        for line in temp:
            file.write(line + "\n")
            
runnet('wnet0.txt', 'testnet0.txt', 'result-nn0.txt')