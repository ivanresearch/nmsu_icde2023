import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def validation(model, testloader, criterion, device='cpu'):
    accuracy = 0
    test_loss = 0
    validation_time = time.time()
    for inputs, labels in testloader:
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(testloader), accuracy/len(testloader), time.time()-validation_time


def model_parameter_debug(model):
    for name, parameters in model.named_parameters():
        if name == "bn3.weight":
            print(name)
            print(parameters[0:4])
        if name == "csa_model.csa_sigma":
            print(name)
            print(parameters)
        if name == "csa_model.query_conv1.bias":
            print(name)
            print(parameters[0:2])

def train(logger, model, trainloader, validloader, train_x_tensor, train_y_tensor, class_id_list, criterion, optimizer, 
          epochs=10, print_every=10, device='cpu', saved_path='../object/default'):
    # print("Training started on device: {}".format(device))
    logger.info("Training started on device: " + str(device))

    valid_loss_min = np.Inf # track change in validation loss
    best_val_accuracy = -1
    saved_path = saved_path + '.pt'

    csa_keyword = 'group_attn'
    print(model.attn_key)
    if csa_keyword in model.attn_key:
        train_x_tensor = train_x_tensor.float()
        train_x_tensor, train_y_tensor = train_x_tensor.to(device), train_y_tensor.to(device)
        for i in range(len(class_id_list)):
            class_id_list[i] = class_id_list[i].to(device)
        model.csa_model.class_id_list = class_id_list
        model.csa_model.store_attn = model.csa_model.store_attn.to(device)
    # train_time = time.time() - start_time
    start_time = time.time()
    for iter_epoch in range(epochs):
        train_loss = 0.0
        csa_train_loss = 0.0
        valid_loss = 0.0
        
        model.train()
        for name, parameters in model.named_parameters():
            if name.startswith('csa_model'):
                parameters.requires_grad = False
            else:
                parameters.requires_grad = True
        # print("Epoch: " + str(iter_epoch) + " Point 1")
        # print(model.csa_model.store_attn.shape)
        # print(model.csa_model.store_attn[0, 0:2, 0:4])
        csa_training = False
        for inputs, labels in trainloader:
            # steps += 1
            inputs = inputs.float()
            inputs, labels = inputs.to(device),labels.to(device)
            
            optimizer.zero_grad()
            output = model.forward(inputs, iter_epoch, csa_training)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            # print("Each iteration")
            # print(model.csa_model.store_attn[0, 0:5, 0:2])
            # print(model.csa_model.key_conv1.bias[0:3])
            # print(model.csa_model.csa_sigma)
            
            train_loss += loss.item()


        if csa_keyword in model.attn_key:
            ## CSA part if
            # print("Start CSA training")
            # print(model.csa_model.store_attn[0, 0:5, 0:2])
            # print(model.csa_model.key_conv1.bias[0:3])
            # print(model.csa_model.csa_sigma)
            csa_training = True
            for name, parameters in model.named_parameters():
                if name.startswith('csa_model'):
                    parameters.requires_grad = True
                else:
                    parameters.requires_grad = False
            optimizer.zero_grad()
            csa_output = model.forward(train_x_tensor, iter_epoch, csa_training)
            csa_loss = criterion(csa_output, train_y_tensor)
            csa_loss.backward()
            optimizer.step()
            model.csa_model.store_attn.detach_()
            csa_train_loss += csa_loss.item()
            # print("End CSA training")
            # print(model.csa_model.store_attn[0, 0:5, 0:2])
            # print(model.csa_model.key_conv1.bias[0:3])
            # print(model.csa_model.csa_sigma)
            ## End of CSA part

        # print("Epoch: " + str(iter_epoch) + " Point 3")
        # print(model.csa_model.store_attn.shape)
        # print(model.csa_model.store_attn[0, 0:2, 0:4])

        # Do evaluation on validation for each epoch
        model.eval()       
        with torch.no_grad():
            valid_loss, val_accuracy, _ = validation(model, validloader, criterion, device)
            # print("Inside: " + str(val_accuracy))
            # print(len(validloader))
        
        logger.info("Epoch {}/{}: Training Loss: {:.6f}.. CSA Training Loss: {:.6f}.. Val Loss: {:.6f}.. Val Accuracy: {:.2f}".format(iter_epoch+1, epochs, train_loss/print_every, csa_train_loss, valid_loss, val_accuracy))
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            valid_str = 'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss)
            logger.info(valid_str)
            torch.save(model.state_dict(), saved_path)
            valid_loss_min = valid_loss
        train_loss = 0
        model.train()
    train_time = time.time() - start_time  
    # print("Before return")
    # with torch.no_grad():
    #     valid_loss, val_accuracy, _ = validation(model, validloader, criterion, device)
    #     print(val_accuracy)
    #     print(len(validloader))
    # model.eval()
    # with torch.no_grad():
    #     valid_loss, val_accuracy, _ = validation(model, validloader, criterion, device)
    #     print("Beginning Inside: " + str(val_accuracy))
    #     print(len(validloader))
    # print("End of Before return")
    # print("End of Before return")
    return best_val_accuracy, saved_path, train_time


def load_datasets(dataset_name='ISLD'):
    data_path = './datasets/'+dataset_name+'/'

    X_train = torch.load(data_path+'X_train_tensor.pt')
    X_val = torch.load(data_path+'X_val_tensor.pt')
    X_test = torch.load(data_path+'X_test_tensor.pt')

    y_train = torch.load(data_path+'y_train_tensor.pt')
    y_val = torch.load(data_path+'y_val_tensor.pt')
    y_test = torch.load(data_path+'y_test_tensor.pt')

    seq_lens_train = torch.load(data_path+'seq_lens_train_tensor.pt')
    seq_lens_val = torch.load(data_path+'seq_lens_val_tensor.pt')
    seq_lens_test = torch.load(data_path+'seq_lens_test_tensor.pt')

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, seq_lens_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val, seq_lens_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test, seq_lens_test)

    return train_dataset, val_dataset, test_dataset
