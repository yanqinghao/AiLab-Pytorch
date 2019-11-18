import numpy as np
import suanpan
from suanpan.app import app
from suanpan.app.arguments import Npy, Json


@app.input(Json(key="inputData"))
@app.output(Npy(key="outputImage"))
def transform(context):
    args = context.args
    return np.array(args.data["images"])


if __name__ == "__main__":
    suanpan.run(app)
