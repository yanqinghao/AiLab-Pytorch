import pandas as pd
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Csv, Json


@app.input(Json(key="inputData1"))
@app.output(Csv(key="outputData1"))
def transform(context):
    args = context.args
    return pd.DataFrame(args.inputData1["text"], columns=["text"])


if __name__ == "__main__":
    suanpan.run(app)
