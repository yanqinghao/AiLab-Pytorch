from suanpan import g
from suanpan.screenshots import Screenshots


def getScreenshotPath():
    node_info = (g.userId, g.appId, g.nodeId)
    return "studio/{}/logs/{}/screenshots/{}".format(*node_info)


def createScreenshots(storage_name):
    screenshots = Screenshots()
    screenshots.current.storageName = storage_name
    screenshots.clean()
    return screenshots
