import numpy as np
from mlp_handwritten_digits.MLPClassifierLAR import MLPClassifier

# Limpa a tela
print("\x1b[2J\x1b[1;1H")

mlp = MLPClassifier(learning_rate=0.001, neurons_hidden_layer=75, error_threshold=10, 
                        max_iterations=10000, num_exemplares=90, lr_increase_factor=1.01, lr_decrease_factor=0.1,
                        min_learning_rate=1e-5, improvement_tolerance=1e-4)

# Em vez de treinar, carregamos os pesos e bias dos arquivos CSV gerados anteriormente.
mlp.weights_input_hidden = np.loadtxt('./result/input_weights_v1.csv', delimiter=',')
mlp.bias_hidden = np.loadtxt('./result/input_bias_v1.csv', delimiter=',').reshape(-1, 1)
mlp.weights_hidden_output = np.loadtxt('./result/output_weights_v1.csv', delimiter=',')
mlp.bias_output = np.loadtxt('./result/output_bias_v1.csv', delimiter=',').reshape(-1, 1)


# Parâmetros para os arquivos de teste:
# Espera-se que os arquivos estejam na pasta "data/" com nome no padrão: "<classe>_<número>.txt"
aminicial = 30         # Número inicial para os nomes dos arquivos de teste
amtestedigitos = 20     # Número de amostras de teste para cada dígito

total_tests = 0
correct_count = 0

# Loop para cada dígito (classe) de 0 a 9 e para as amostras de teste
for digit in range(10):
    for n in range(amtestedigitos):
        sample_number = n + aminicial
        filename = f"data/{digit}_{sample_number}.txt"
        try:
            xtest = np.loadtxt(filename)
        except Exception as e:
            print(f"Erro ao carregar {filename}: {e}")
            continue
        
        # Executa a predição para a amostra
        y_pred = mlp.predict(xtest)
        # A classe prevista é definida pelo índice do maior valor (deve ser 1 após limiarização)
        predicted_digit = np.argmax(y_pred)
        
        if predicted_digit == digit:
            correct_count += 1
        total_tests += 1

# Cálculo da acurácia
accuracy = correct_count / total_tests if total_tests > 0 else 0.0
print(f"\nAcurácia total nos dados de teste: {accuracy:.4f}")

