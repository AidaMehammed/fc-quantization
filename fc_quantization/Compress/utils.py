import torch
import copy
import os
import torch.nn as nn




def get_weights(model):
    return [param.data for param in model.parameters()]

def set_weights(model, weights):
  for param, weight in zip(model.parameters(), weights):
        param.data = weight

def prepare_qat(model,qconfig):
    qmodel = copy.deepcopy(model)

    torch.backends.quantized.engine = qconfig
    qmodel.qconfig = torch.quantization.get_default_qconfig(qconfig)

    qmodel.train()

    temp = torch.quantization.prepare_qat(qmodel, inplace=False)

    return temp


def prep_post_static_quant(model, train_loader, qconfig):
    qmodel = copy.deepcopy(model)

    torch.backends.quantized.engine = qconfig
    qmodel.qconfig = torch.quantization.get_default_qconfig(qconfig)

    temp = torch.quantization.prepare(qmodel, inplace=False)
    temp.eval()

    # Calibrate with the training data
    with torch.no_grad():
        for data, target in train_loader:
            # temp(data)
            forward_quant(temp, data)

    return temp

# essential for training and testing functions
def forward_quant(model, x):
    # Add quantization stub
    x = model.quant(x)

    # Forward pass through the model
    x = model(x)

    # Add dequantization stub
    x = model.dequant(x)

    return x



def revert_quantized_model(quantized_model, original_model):

    reverted_model = nn.Sequential()

    # Iterate through the named parameters of the original model
    for name, param in original_model.named_parameters():
        # Extract the parameter name without the "weight" suffix
        param_name, param_type = name.rsplit('.', 1)

        if param_type == 'weight':
            quantized_param = getattr(quantized_model, param_name).weight()
        else:
            quantized_param = getattr(quantized_model, param_name).bias()

        dequantized_param = quantized_param.dequantize()

        name = name.replace('.', '_')

        setattr(reverted_model, name, nn.Parameter(dequantized_param))


    return reverted_model




def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6  # size in MB
    os.remove('temp.p')
    return size


