import torch

def predict(model, loader, device):
    predictions = []
    actual = []
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            output = model(data)
            predictions.append(output.cpu().numpy())
            actual.append(target.cpu().numpy())
    return predictions, actual