# coding=utf-8
from __future__ import absolute_import, print_function

import os
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Int, Folder, String, ListOfString
from utils import text_classification
from args import PytorchDataset


def find_all_files(folder):
    files_ = []
    list = [i for i in os.listdir(folder)]
    for i in range(0, len(list)):
        path = os.path.join(folder, list[i])
        if os.path.isdir(path):
            files_.extend(find_all_files(path))
        if not os.path.isdir(path):
            if path.lower().endswith("csv"):
                files_.append(path)
    return files_


@app.input(Folder(key="inputData"))
@app.param(ListOfString(key="featureColumns"))
@app.param(String(key="labelColumn"))
@app.param(String(key="dataSets", default="AG_NEWS"))
@app.param(Int(key="NGRAMS", default=2))
@app.output(PytorchDataset(key="outputTrainData"))
@app.output(PytorchDataset(key="outputTestData"))
def SPTextDataset(context):
    args = context.args
    NGRAMS = args.NGRAMS
    dataSets = args.dataSets
    if dataSets != "PRED_Data":
        train_dataset, test_dataset = text_classification.DATASETS[dataSets](find_all_files(
            args.inputData),
                                                                             root="./.data",
                                                                             ngrams=NGRAMS,
                                                                             vocab=None)
        setattr(train_dataset, "collate", "generate_batch")
        setattr(test_dataset, "collate", "generate_batch")
        setattr(
            train_dataset,
            "class_to_idx",
            {
                "World": 1,
                "Sports": 2,
                "Business": 3,
                "Sci/Tec": 4
            },
        )
        setattr(
            test_dataset,
            "class_to_idx",
            {
                "World": 1,
                "Sports": 2,
                "Business": 3,
                "Sci/Tec": 4
            },
        )
        setattr(train_dataset, "NGRAMS", NGRAMS)
        setattr(test_dataset, "NGRAMS", NGRAMS)
        return train_dataset, test_dataset
    else:
        pred_dataset = text_classification.DATASETS[dataSets](
            find_all_files(args.inputData)[0],
            args.featureColumns,
            label=args.labelColumn,
        )
        setattr(pred_dataset, "collate", "generate_batch")
        return pred_dataset, None


if __name__ == "__main__":
    suanpan.run(app)
