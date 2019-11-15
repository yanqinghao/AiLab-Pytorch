import torch


def generate_batch(batch):
    label = (
        torch.tensor([entry[0] for entry in batch])
        if batch[0][0]
        else [entry[0] for entry in batch]
    )
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    idxs = torch.tensor([entry[2] for entry in batch])
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return {"input": text, "offsets": offsets}, label, idxs
