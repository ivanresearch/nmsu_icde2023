import numpy as np

def train_test_loading(path, type='train'):
    x_matrix = np.load(path + 'X_' + type + '.npy')
    y_vector = np.load(path + 'y_' + type +'.npy').astype(int)
    nclass = int(np.amax(y_vector)) + 1
    return x_matrix, y_vector, nclass

def data_reading(data_key):
    data_folder = "../mts_data/" + data_key + '/'
    
    type = 'train'
    train_x_matrix, train_y_vector, nclass = train_test_loading(data_folder, type)
    print("Training data")
    print(train_x_matrix.shape, train_y_vector.shape, nclass)
    test_x_matrix, test_y_vector, nclass = train_test_loading(data_folder, type)
    print("Testing data")
    print(test_x_matrix.shape, test_y_vector.shape, nclass)

if __name__ == '__main__':
    data_key = 'Cricket'
    data_reading(data_key)
    
