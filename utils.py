import json


def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
