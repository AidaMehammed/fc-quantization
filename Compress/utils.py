import torch
import copy
import os
import torch.nn as nn
import torch.optim as optim


def print_model_shapes(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Shape: {param.shape}")

def get_weights(model):
    return [param.data for param in model.parameters()]

def set_weights(model, weights):
  for param, weight in zip(model.parameters(), weights):
        param.data = weight



def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


def train(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = forward_quant(model, data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {100 * train_accuracy:.2f}%, '
              )
    print('Finished Training')


def forward_quant(model, x):
    # Add quantization stub
    x = model.quant(x)

    # Forward pass through the model
    x = model(x)

    # Add dequantization stub
    x = model.dequant(x)

    return x


def apply_qat(model, train_loader,qconfig, epochs, lr):
    qmodel = copy.deepcopy(model)
    torch.backends.quantized.engine = qconfig
    qmodel.qconfig = torch.quantization.get_default_qconfig(qconfig)
    qmodel.train()

    temp = torch.quantization.prepare_qat(qmodel, inplace=False)

    print("Training QAT Model...")
    train(temp, train_loader, epochs, lr)

    qmodel = torch.quantization.convert(temp, inplace=False)
    return qmodel




def post_static_quant(model, train_loader, qconfig):
    qmodel = copy.deepcopy(model)

    torch.backends.quantized.engine = qconfig
    qmodel.qconfig = torch.quantization.get_default_qconfig(qconfig)

    temp = torch.quantization.prepare(qmodel, inplace=False)
    temp.eval()

    # Calibrate with the training data
    with torch.no_grad():
        for data, target in train_loader:
            forward_quant(temp, data)

    qmodel = torch.quantization.convert(temp, inplace=False)
    return qmodel




def revert_quantized_model(quantized_model, original_model):

    reverted_model = nn.Sequential()

    # Iterate through the named parameters of the original model
    for name, param in original_model.named_parameters():
        param_name, param_type = name.rsplit('.', 1)

        if param_type == 'weight':
            quantized_param = getattr(quantized_model, param_name).weight()
        else:
            quantized_param = getattr(quantized_model, param_name).bias()

        dequantized_param = quantized_param.dequantize()

        name = name.replace('.', '_')

        setattr(reverted_model, name, nn.Parameter(dequantized_param))


    return reverted_model





def average_weights_quant(reconstructed_clients):
    num_clients = len(reconstructed_clients)
    num_weights = len(reconstructed_clients[0])

    summed_weights = []
    for i in range(num_weights):
        total_weight = sum(weights[i] for weights in reconstructed_clients)
        summed_weights.append(total_weight)

    averaged_weights = []
    for weight_sum in summed_weights:
        averaged_weight = weight_sum / num_clients
        averaged_weights.append(averaged_weight)

    print('Data has been averaged')

    return averaged_weights


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # size in MB
    os.remove('temp.p')
    return size


