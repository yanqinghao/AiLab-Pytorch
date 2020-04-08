import numpy as np
import suanpan
from suanpan import g
from suanpan.app import app
from suanpan.app.arguments import Npy, Json
from suanpan.utils import image
from suanpan.storage import storage


@app.input(Json(key="inputData1"))
@app.output(Npy(key="outputData1"))
def transform(context):
    args = context.args
    channels = args.inputData1.get("channels", 1)
    images = []
    for path in args.inputData1["images"]:
        imagePath = storage.getPathInTempStore(
            storage.storagePathJoin(
                "studio", g.userId, g.appId, "-".join(path.split(storage.delimiter))
            )
        )
        storage.download(path, imagePath)
        images.append(image.read(imagePath, channels))
    return np.array(images)


if __name__ == "__main__":
    suanpan.run(app)
