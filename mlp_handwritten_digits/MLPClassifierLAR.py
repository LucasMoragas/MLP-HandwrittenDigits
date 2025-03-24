import numpy as np
from mlp_handwritten_digits.get_data import get_train_data

class MLPClassifier:
    def __init__(self, learning_rate=0.001, neurons_hidden_layer=50, error_threshold=0.001, 
                 max_iterations=10000, num_exemplares=90, lr_increase_factor=1.05, lr_decrease_factor=0.7,
                 min_learning_rate=1e-6, improvement_tolerance=1e-5):
        
        self.X = get_train_data()
        self.input_dim = 256 
        self.output_dim = 10
        self.learning_rate = learning_rate
        self.lr_increase_factor = lr_increase_factor
        self.lr_decrease_factor = lr_decrease_factor
        self.min_learning_rate = min_learning_rate
        self.improvement_tolerance = improvement_tolerance
        self.neurons_hidden_layer = neurons_hidden_layer
        self.error_threshold = error_threshold
        self.max_iterations = max_iterations

        self.T = self.create_targets(num_exemplares)

        self.weights_input_hidden = np.random.rand(self.neurons_hidden_layer, self.input_dim) - 0.5
        self.bias_hidden = np.random.rand(self.neurons_hidden_layer, 1) - 0.5
        self.weights_hidden_output = np.random.rand(self.output_dim, self.neurons_hidden_layer) - 0.5
        self.bias_output = np.random.rand(self.output_dim, 1) - 0.5

        self.error = float('inf')
        self.iterations = 0
        self.error_history = []
        self.iterations_history = []
        self.training_complete = False

    def create_targets(self, num_exemplares):
        n_amostras = num_exemplares * 10
        targets = np.empty((n_amostras, 10))
        for i in range(n_amostras):
            t = -np.ones(10)
            digit = i % 10
            t[digit] = 1
            targets[i] = t
        return targets

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2

    def bipolar_step(self, x):
        return np.where(x >= 0, 1, -1)

    def train(self):
        n_amostras = self.X.shape[0]
        prev_epoch_error = float('inf')
        
        while self.error > self.error_threshold and self.iterations < self.max_iterations:
            self.iterations += 1
            epoch_error = 0

            for i in range(n_amostras):
                x_sample = self.X[i].reshape(self.input_dim, 1)
                t_sample = self.T[i].reshape(self.output_dim, 1)

                z_hidden = np.dot(self.weights_input_hidden, x_sample) + self.bias_hidden
                a_hidden = self.tanh(z_hidden)

                z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
                a_output = self.tanh(z_output)

                sample_error = 0.5 * np.sum((t_sample - a_output)**2)
                epoch_error += sample_error

                delta_output = (t_sample - a_output) * self.tanh_derivative(z_output)
                delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * self.tanh_derivative(z_hidden)

                self.weights_hidden_output += self.learning_rate * np.dot(delta_output, a_hidden.T)
                self.bias_output += self.learning_rate * delta_output
                self.weights_input_hidden += self.learning_rate * np.dot(delta_hidden, x_sample.T)
                self.bias_hidden += self.learning_rate * delta_hidden

            self.error_history.append(epoch_error)
            self.iterations_history.append(self.iterations)
            self.error = epoch_error

            # Verifica se houve melhora significativa
            if (prev_epoch_error - epoch_error) > self.improvement_tolerance:
                # Melhora significativa: aumenta a taxa de aprendizagem
                self.learning_rate *= self.lr_increase_factor
            else:
                # Se não houve melhora significativa, diminui a taxa
                self.learning_rate *= self.lr_decrease_factor
                # Garante que a taxa não caia abaixo do mínimo
                if self.learning_rate < self.min_learning_rate:
                    self.learning_rate = self.min_learning_rate

            prev_epoch_error = epoch_error

            if self.iterations % 100 == 0:
                print(f"Época: {self.iterations}, Erro total: {epoch_error:.6f}, Taxa de aprendizagem: {self.learning_rate:.6f}")

        self.training_complete = True
        print("Treinamento concluído!")

    def predict(self, x):
        x_sample = x.reshape(self.input_dim, 1)
        z_hidden = np.dot(self.weights_input_hidden, x_sample) + self.bias_hidden
        a_hidden = self.tanh(z_hidden)
        z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
        a_output = self.tanh(z_output)
        return self.bipolar_step(a_output)

    def compute_accuracy(self):
        n_amostras = self.X.shape[0]
        correct = 0
        for i in range(n_amostras):
            prediction = self.predict(self.X[i])
            predicted_class = np.argmax(prediction)
            true_class = np.argmax(self.T[i])
            if predicted_class == true_class:
                correct += 1
        return correct / n_amostras
    
    def generate_weights_and_bias(self):
        version = 3
        
        np.savetxt(f'./result/input_weights_v{version}.csv', self.weights_input_hidden, fmt='%f', delimiter=',')
        np.savetxt(f'./result/input_bias_v{version}.csv', self.bias_hidden, fmt='%f', delimiter=',')
        np.savetxt(f'./result/output_weights_v{version}.csv', self.weights_hidden_output, fmt='%f', delimiter=',')
        np.savetxt(f'./result/output_bias_v{version}.csv', self.bias_output, fmt='%f', delimiter=',')


if __name__ == '__main__':
    mlp = MLPClassifier(learning_rate=0.001, neurons_hidden_layer=75, error_threshold=8, 
                        max_iterations=10000, num_exemplares=90, lr_increase_factor=1.005, lr_decrease_factor=0.05,
                        min_learning_rate=1e-5, improvement_tolerance=1e-4)
    
    mlp.train()
    
    acc = mlp.compute_accuracy()
    print("\nAcurácia total:", acc)
    
    mlp.generate_weights_and_bias()
