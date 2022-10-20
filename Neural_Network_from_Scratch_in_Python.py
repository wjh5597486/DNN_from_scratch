
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

"""
DNN from Scratch
"""

np.random.seed(222)

class nnlayer:
    def __init__(self, input_num, output_num):
        self.weights = np.random.randn(input_num, output_num) * 0.1
        self.biases = np.zeros((1, output_num))

    def forward(self, values):
        self.inputs = values
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, values):
        self.dweight = np.dot(self.inputs.T, values)
        self.dbiases = np.sum(values, axis=0, keepdims=True)
        self.dinputs = np.dot(values, self.weights.T)


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Sigmoid:
    def forward(self, dinputs):
        self.output = 1 / (1 + np.exp(-dinputs))

    def backward(self, values):
        self.dinputs = values.copy()
        self.dinputs = self.dinputs * (1.0 - self.dinputs)


class SoftMax:
    def forward(self, values):
        self.inputs = values
        exp_values = np.exp(self.inputs - np.max(values, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class NeuralNetwork:
    def __init__(self):
        self.output = None
        self.layer1 = nnlayer(784, 100)
        self.activation1 = ReLU()
        self.layer2 = nnlayer(100, 100)
        self.activation2 = ReLU()
        self.layer3 = nnlayer(100, 10)

    def forward(self, value):
        self.layer1.forward(value)
        self.activation1.forward(self.layer1.output)
        self.layer2.forward(self.activation1.output)
        self.activation2.forward(self.layer2.output)
        self.layer3.forward(self.activation2.output)
        self.output = self.layer3.output

    def backward(self, value):
        self.dinputs = value
        self.layer3.backward(self.dinputs)
        self.activation2.backward(self.layer3.dinputs)
        self.layer2.backward(self.activation2.dinputs)
        self.activation1.backward(self.layer2.dinputs)
        self.layer1.backward(self.activation1.dinputs)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):  # https://www.youtube.com/watch?v=levekYbxauw
    def forward(self, y_pred, y_true):

        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)  # set MIN/MAX

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2: #정답이 2개???
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy(): # Y-y > return : MSE
    def __init__(self):
        self.activation = SoftMax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)  # getting the number of classes
        if len(y_true.shape) == 2:  # classification: [0.7 0.3] => [1]
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweight
        layer.biases += -self.learning_rate * layer.dbiases


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = DataLoader(dataset=mnist_train,
                          batch_size=batch_size, # 배치 크기는 100
                          shuffle=True,
                          drop_last=True)

# Optimizer
model = NeuralNetwork()
optimizer = Optimizer_SGD()
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

epoch = 5

for i in range(epoch):
    print(f'epoch:{i+1}')
    for X, Y in data_loader:

        X = X.view(-1, 28 * 28)

        # Propagation
        model.forward(X)

        # Cost Function
        loss = loss_activation.forward(model.output, Y)
        # Backpropagation.

        loss_activation.backward(loss_activation.output, Y)
        model.backward(loss_activation.dinputs)

        # Update Weights and Biases
        optimizer.update_params(model.layer1)
        optimizer.update_params(model.layer2)
        optimizer.update_params(model.layer3)

    with torch.no_grad():
        X_test = mnist_test.test_data.view(-1, 28 * 28)
        Y_test = mnist_test.test_labels
        model.forward(X_test)
        prediction = model.output
        prediction = torch.Tensor(prediction)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
    model.forward(torch.zeros_like(X_test))


# Print
# print(model.output)
# y_pred = np.argmax(model.output, axis=1)
# print('acc : ', np.mean(y_goal == y_pred))
