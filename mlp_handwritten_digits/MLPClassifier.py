import numpy as np
from mlp_handwritten_digits.get_data import get_train_data

class MLPClassifier:
    def __init__(self, learning_rate=0.001, neurons_hidden_layer=50, error_threshold=0.001, max_iterations=10000, num_exemplares=90):
        """
        Inicializa os parâmetros do modelo para classificação.

        Parâmetros:
          - learning_rate: taxa de aprendizado.
          - neurons_hidden_layer: número de neurônios na camada oculta.
          - error_threshold: erro mínimo para término do treinamento.
          - max_iterations: número máximo de iterações para evitar loop infinito.
          - num_exemplares: número de exemplares para cada classe (dígitos 0-9).
                           Total de amostras será num_exemplares * 10 (neste caso, 90*10 = 900).

        Observações:
          - A dimensão de entrada é determinada a partir dos dados carregados (deve ser 255 conforme o get_train_data).
          - A dimensão de saída é 10 (dígitos de 0 a 9) com codificação bipolar.
          - self.X é carregado a partir do arquivo txt.
          - self.T é gerado automaticamente com base na ordem dos dados.
        """
        # Carrega os dados de treinamento (X terá shape [900, 255])
        self.X = get_train_data()
        self.input_dim = self.X.shape[1]  # Ajusta dinamicamente com base nos dados (espera-se 255)
        self.output_dim = 10
        self.learning_rate = learning_rate
        self.neurons_hidden_layer = neurons_hidden_layer
        self.error_threshold = error_threshold
        self.max_iterations = max_iterations

        # Gera a matriz de targets (T) de acordo com a ordem dos dados:
        # a linha 1 é dígito 0, linha 2 dígito 1, ..., linha 10 dígito 9, linha 11 dígito 0, etc.
        self.T = self.create_targets(num_exemplares)

        # Inicializa os pesos e bias com valores aleatórios pequenos (centralizados em zero)
        self.weights_input_hidden = np.random.rand(self.neurons_hidden_layer, self.input_dim) - 0.5
        self.bias_hidden = np.random.rand(self.neurons_hidden_layer, 1) - 0.5
        self.weights_hidden_output = np.random.rand(self.output_dim, self.neurons_hidden_layer) - 0.5
        self.bias_output = np.random.rand(self.output_dim, 1) - 0.5

        # Variáveis de controle do treinamento
        self.error = float('inf')
        self.iterations = 0
        self.error_history = []
        self.iterations_history = []
        self.training_complete = False

    def create_targets(self, num_exemplares):
        """
        Cria a matriz de targets para classificação de acordo com a ordem dos dados.

        Considerando que:
          - A linha 1 (índice 0) corresponde ao dígito 0,
          - A linha 2 (índice 1) corresponde ao dígito 1,
          - ...
          - A linha 10 (índice 9) corresponde ao dígito 9,
          - A linha 11 (índice 10) corresponde novamente ao dígito 0, e assim sucessivamente,

        Este método gera uma matriz com shape (num_exemplares*10, 10), onde para cada amostra i:
          - É colocado 1 na posição (i mod 10);
          - São colocados -1 nas demais posições.
        """
        n_amostras = num_exemplares * 10
        targets = np.empty((n_amostras, 10))
        for i in range(n_amostras):
            t = -np.ones(10)
            digit = i % 10  # Determina o dígito correspondente à linha
            t[digit] = 1
            targets[i] = t
        return targets

    def tanh(self, x):
        """Função de ativação tangente hiperbólica."""
        return np.tanh(x)

    def tanh_derivative(self, x):
        """Derivada da função tangente hiperbólica."""
        return 1 - np.tanh(x)**2

    def bipolar_step(self, x):
        """
        Função degrau bipolar.
        Converte cada elemento de x para 1 se for >= 0 ou para -1 caso contrário.
        """
        return np.where(x >= 0, 1, -1)

    def train(self):
        """
        Executa o treinamento da MLP utilizando o algoritmo de backpropagation.

        Para cada amostra:
          - Realiza o forward pass (camada oculta e camada de saída);
          - Calcula o erro quadrático em relação ao target;
          - Executa o backpropagation para ajustar os pesos e bias.

        O treinamento para quando o erro total acumulado na iteração for menor que error_threshold
        ou quando o número máximo de iterações for atingido.
        """
        n_amostras = self.X.shape[0]

        while self.error > self.error_threshold and self.iterations < self.max_iterations:
            self.iterations += 1
            self.error = 0

            for i in range(n_amostras):
                # Converte a amostra e o target em vetores coluna
                x_sample = self.X[i].reshape(self.input_dim, 1)
                t_sample = self.T[i].reshape(self.output_dim, 1)

                # Forward pass: camada oculta
                z_hidden = np.dot(self.weights_input_hidden, x_sample) + self.bias_hidden
                a_hidden = self.tanh(z_hidden)

                # Forward pass: camada de saída
                z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
                a_output = self.tanh(z_output)

                # Calcula o erro quadrático para a amostra
                sample_error = 0.5 * np.sum((t_sample - a_output)**2)
                self.error += sample_error

                # Backpropagation: calcula os deltas para a saída e para a camada oculta
                delta_output = (t_sample - a_output) * self.tanh_derivative(z_output)
                delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * self.tanh_derivative(z_hidden)

                # Atualiza os pesos e bias
                self.weights_hidden_output += self.learning_rate * np.dot(delta_output, a_hidden.T)
                self.bias_output += self.learning_rate * delta_output
                self.weights_input_hidden += self.learning_rate * np.dot(delta_hidden, x_sample.T)
                self.bias_hidden += self.learning_rate * delta_hidden

            self.error_history.append(self.error)
            self.iterations_history.append(self.iterations)

            if self.iterations % 100 == 0:
                print(f"Iteração: {self.iterations}, Erro total: {self.error}")

        self.training_complete = True
        print("Treinamento concluído!")

    def predict(self, x):
        """
        Realiza a predição para uma única amostra de entrada.

        Parâmetro:
          - x: vetor de entrada com dimensão igual a self.input_dim (deve ser 255, conforme os dados).

        Retorna:
          - vetor de 10 elementos com valores 1 ou -1, obtidos aplicando o degrau bipolar à saída.
        """
        x_sample = x.reshape(self.input_dim, 1)
        z_hidden = np.dot(self.weights_input_hidden, x_sample) + self.bias_hidden
        a_hidden = self.tanh(z_hidden)
        z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
        a_output = self.tanh(z_output)
        return self.bipolar_step(a_output)

    def compute_accuracy(self):
        """
        Executa a predição para todas as amostras e calcula a acurácia,
        ou seja, a razão entre o número de amostras classificadas corretamente e o total de amostras.
        A classificação é definida pelo índice do maior valor na saída (argmax).
        """
        n_amostras = self.X.shape[0]
        correct = 0
        for i in range(n_amostras):
            # Obtém a predição para a amostra i
            prediction = self.predict(self.X[i])
            # A classe prevista é o índice do maior valor na predição
            predicted_class = np.argmax(prediction)
            # A classe verdadeira é o índice onde o target possui o valor 1
            true_class = np.argmax(self.T[i])
            if predicted_class == true_class:
                correct += 1
        accuracy = correct / n_amostras
        return accuracy

if __name__ == '__main__':
    # Cria uma instância da MLP com os parâmetros desejados
    mlp = MLPClassifier(learning_rate=0.001, neurons_hidden_layer=50, error_threshold=0.001, max_iterations=10000, num_exemplares=90)
    
    # Inicia o treinamento
    mlp.train()
    
    # Calcula e imprime a acurácia total sobre todas as amostras
    acc = mlp.compute_accuracy()
    print("\nAcurácia total:", acc)
