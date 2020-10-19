import torch


def load_model(file_path):
    with open(file_path, "rb") as f:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.load(f, map_location=device)


def dump_model(value, file_path):
    with open(file_path, "wb") as f:
        torch.save(value, f)
