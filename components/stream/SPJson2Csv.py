import pandas as pd
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Csv, Json


@app.input(Json(key="inputData"))
@app.output(Csv(key="outputCsv"))
def transform(context):
    args = context.args
    return pd.DataFrame(args.data["text"], columns=["text"])


if __name__ == "__main__":
    suanpan.run(app)
