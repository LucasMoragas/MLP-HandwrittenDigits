import numpy as np

class MLPClassifier:
    def __init__(self, learning_rate=0.001, neurons_hidden_layer=50, error_threshold=0.001, max_iterations=10000, num_exemplares=10):
        """
        Inicializa os parâmetros do modelo para classificação.

        Parâmetros:
          - learning_rate: taxa de aprendizado.
          - neurons_hidden_layer: número de neurônios na camada oculta.
          - error_threshold: erro mínimo para término do treinamento.
          - max_iterations: número máximo de iterações para evitar loop infinito.
          - num_exemplares: número de exemplares para cada classe (dígitos 0-9). Total de amostras será num_exemplares*10.
        
        Observações:
          - A dimensão de entrada é 256 (imagens 16x16).
          - A dimensão de saída é 10 (dígitos de 0 a 9) com codificação bipolar.
          - self.X deve ser carregado posteriormente a partir do arquivo txt.
          - self.T é gerado automaticamente com base no vetor target fornecido.
        """
        self.input_dim = 256
        self.output_dim = 10
        self.learning_rate = learning_rate
        self.neurons_hidden_layer = neurons_hidden_layer
        self.error_threshold = error_threshold
        self.max_iterations = max_iterations

        # self.X deve ser carregado posteriormente com os dados de treinamento (shape: [num_amostras, 256])
        self.X = None

        # Gera a matriz de targets (Y) para 10 exemplares de cada dígito (0-9)
        self.T = self.create_targets(num_exemplares)

        # Inicializa os pesos e bias com valores aleatórios pequenos (centralizados em zero)
        self.weights_input_hidden = np.random.rand(self.neurons_hidden_layer, self.input_dim) - 0.5
        self.bias_hidden = np.random.rand(self.neurons_hidden_layer, 1) - 0.5
        self.weights_hidden_output = np.random.rand(self.output_dim, self.neurons_hidden_layer) - 0.5
        self.bias_output = np.random.rand(self.output_dim, 1) - 0.5

        # Inicializa as variáveis de controle do treinamento
        self.error = float('inf')
        self.iterations = 0
        self.error_history = []
        self.iterations_history = []
        self.training_complete = False

    def create_targets(self, num_exemplares):
        """
        Cria a matriz de targets para classificação.
        
        Para cada dígito de 0 a 9, cria um vetor target com:
          - 1 na posição correspondente à classe,
          - -1 nas demais posições.
        
        Cada vetor é repetido num_exemplares vezes, gerando uma matriz com shape (num_exemplares*10, 10).

        Exemplo:
          Para num_exemplares = 10, teremos 100 linhas, onde:
            - A linha 1 representa o target para o dígito 0: [ 1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ]
            - A linha 2 representa o target para o dígito 1: [ -1, 1, -1, -1, -1, -1, -1, -1, -1, -1 ]
            - ...
            - A linha 10 representa o target para o dígito 9: [ -1, -1, -1, -1, -1, -1, -1, -1, -1, 1 ]
            - A linha 11 recomeça para o dígito 0 e assim sucessivamente.
        """
        num_classes = 10
        # Cria uma matriz base de shape (10, 10) com todos os elementos -1
        base_targets = -np.ones((num_classes, num_classes))
        # Define 1 na posição correspondente para cada dígito
        for i in range(num_classes):
            base_targets[i, i] = 1
        # Repete cada vetor target num_exemplares vezes
        targets = np.repeat(base_targets, num_exemplares, axis=0)
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
        if self.X is None:
            raise ValueError("Os dados de entrada (self.X) não foram carregados.")

        n_amostras = self.X.shape[0]

        while self.error > self.error_threshold and self.iterations < self.max_iterations:
            self.iterations += 1
            self.error = 0

            for i in range(n_amostras):
                # Prepara a entrada e o target (converte para vetor coluna)
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

                # Backpropagation: cálculo dos deltas para saída e camada oculta
                delta_output = (t_sample - a_output) * self.tanh_derivative(z_output)
                delta_hidden = np.dot(self.weights_hidden_output.T, delta_output) * self.tanh_derivative(z_hidden)

                # Atualiza os pesos e bias
                self.weights_hidden_output += self.learning_rate * np.dot(delta_output, a_hidden.T)
                self.bias_output += self.learning_rate * delta_output
                self.weights_input_hidden += self.learning_rate * np.dot(delta_hidden, x_sample.T)
                self.bias_hidden += self.learning_rate * delta_hidden

            # Armazena os históricos de erro e iterações para análise
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
          - x: vetor de entrada com 256 elementos (imagem 16x16).
          
        Retorna:
          - vetor de 10 elementos com valores 1 ou -1, obtidos aplicando o degrau bipolar à saída.
        """
        x_sample = x.reshape(self.input_dim, 1)
        z_hidden = np.dot(self.weights_input_hidden, x_sample) + self.bias_hidden
        a_hidden = self.tanh(z_hidden)
        z_output = np.dot(self.weights_hidden_output, a_hidden) + self.bias_output
        a_output = self.tanh(z_output)
        return self.bipolar_step(a_output)
