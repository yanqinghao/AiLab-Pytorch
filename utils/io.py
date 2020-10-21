import torch


def load_model(file_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.load(file_path, map_location=device)


def dump_model(value, file_path):
    torch.save(value, file_path)
