import numpy as np
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Npy, Json


@app.input(Json(key="inputData1"))
@app.output(Npy(key="outputData1"))
def transform(context):
    args = context.args
    return np.array(args.inputData1["images"])


if __name__ == "__main__":
    suanpan.run(app)
