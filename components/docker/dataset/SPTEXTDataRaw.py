# coding=utf-8
from __future__ import absolute_import, print_function

from suanpan.app import app
from suanpan.docker.arguments import Folder, String
from utils import downloadTextDataset


@app.param(
    String(
        key="datasetName",
        default="AG_NEWS",
        help="AG_NEWS', 'SogouNews', 'DBpedia', 'YelpReviewPolarity', 'YelpReviewFull', 'YahooAnswers', 'AmazonReviewPolarity', 'AmazonReviewFull'",
    )
)
@app.param(String(key="storageType", default="oss"))
@app.output(Folder(key="outputDir"))
def SPTEXTDataRaw(context):
    args = context.args

    downloadTextDataset(args.datasetName, args.storageType, root=args.outputDir)

    return args.outputDir


if __name__ == "__main__":
    SPTEXTDataRaw()  # pylint: disable=no-value-for-parameter
