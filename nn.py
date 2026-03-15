import numpy as np

class NeuralNetwork:

    def __init__(self, input_size, hidden_architecture, hidden_activation, output_activation):
        # guarda as infos da rede
        self.input_size = input_size
        self.hidden_architecture = hidden_architecture
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def compute_num_weights(self):
        # conta os pesos da rede 
        total_weights = 0
        size = self.input_size

        for layer_size in self.hidden_architecture:
            total_weights += (size + 1) * layer_size  # +1 por causa do bias
            size = layer_size

        total_weights += size + 1  # camada de saída
        return total_weights

    def load_weights(self, weights):
        # recebe os pesos como uma lista 1D e divide por camadas
        w = np.array(weights)

        self.hidden_weights = []
        self.hidden_biases = []

        i = 0
        size = self.input_size

        for layer_size in self.hidden_architecture:
            num_weights = (size + 1) * layer_size
            layer_wb = w[i:i + num_weights].reshape(layer_size, size + 1)

            W = layer_wb[:, :-1]  # pesos
            b = layer_wb[:, -1]   # bias

            self.hidden_weights.append(W.T)
            self.hidden_biases.append(b)

            i += num_weights
            size = layer_size

        # pesos e bias da camada final (saida)
        self.output_weights = w[i:i + size]
        self.output_bias = w[i + size]

    def forward(self, x):
        # da run a rede entrada -> saída
        a = x
        for W, b in zip(self.hidden_weights, self.hidden_biases):
            z = np.dot(a, W) + b
            a = self.hidden_activation(z)

        saida = np.dot(a, self.output_weights) + self.output_bias
        return self.output_activation(saida)


def create_network_architecture(input_size):
    # sigmoide 
    hidden_fn = lambda x: 1 / (1 + np.exp(-x))

    # se for postivio devolve 1 se for negativo -1
    output_fn = lambda x: 1 if x > 0 else -1

    # cria a rede 
    return NeuralNetwork(input_size, (), hidden_fn, output_fn)
