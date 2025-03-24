import numpy as np
from mlp_handwritten_digits.MLPClassifierLAR import MLPClassifier

mlp = MLPClassifier(learning_rate=0.001, neurons_hidden_layer=75, error_threshold=10, 
                        max_iterations=10000, num_exemplares=90, lr_increase_factor=1.01, lr_decrease_factor=0.1,
                        min_learning_rate=1e-5, improvement_tolerance=1e-4)

version = 1

mlp.weights_input_hidden = np.loadtxt(f'./result/input_weights_v{version}.csv', delimiter=',')
mlp.bias_hidden = np.loadtxt(f'./result/input_bias_v{version}.csv', delimiter=',').reshape(-1, 1)
mlp.weights_hidden_output = np.loadtxt(f'./result/output_weights_v{version}.csv', delimiter=',')
mlp.bias_output = np.loadtxt(f'./result/output_bias_v{version}.csv', delimiter=',').reshape(-1, 1)

aminicial = 30         # Número inicial para os nomes dos arquivos de teste
amtestedigitos = 20     # Número de amostras de teste para cada dígito

total_tests = 0
correct_count = 0

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

