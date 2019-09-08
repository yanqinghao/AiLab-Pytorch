from suanpan.screenshots import Screenshots


def createScreenshots(storage_name):
    screenshots = Screenshots()
    screenshots.current.storageName = storage_name
    screenshots.clean()
    return screenshots
