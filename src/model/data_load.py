import torch

def torch_datasets(data_group):
    train_x_matrix = data_group.train_x_matrix
    train_y_vector = data_group.train_y_vector
    test_x_matrix = data_group.test_x_matrix
    test_y_vector = data_group.test_y_vector

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_x_matrix), torch.tensor(train_y_vector))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x_matrix), torch.tensor(test_y_vector))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_x_matrix), torch.tensor(test_y_vector))
    return train_dataset, valid_dataset, test_dataset


def torch_dataloader(data_group, batch_size):
    train_dataset, valid_dataset, test_dataset = torch_datasets(data_group)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def train_tensorloader(data_group):
    train_x_tensor = torch.FloatTensor(data_group.train_x_matrix)
    train_y_tensor = torch.LongTensor(data_group.train_y_vector)
    return train_x_tensor, train_y_tensor