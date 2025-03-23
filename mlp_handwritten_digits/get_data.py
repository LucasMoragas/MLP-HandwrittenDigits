import numpy as np

def get_train_data():
    data = np.loadtxt('./data/digitostreinamento900.txt')
    # Como o arquivo possui 256 colunas e você deseja descartar a última (caso seja extraneous),
    # o resultado terá shape [900, 255]. Caso contrário, remova o slicing.
    X = data[:, :-1]
    return X

if __name__ == '__main__':
    X = get_train_data()
    print("Shape dos dados:", X.shape)
