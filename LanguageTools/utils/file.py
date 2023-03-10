import json


def write_mapping_to_json(mapping, path):
    json_string = json.dumps(mapping, indent=4)
    with open(path, "w") as sink:
        sink.write(json_string)


def read_mapping_from_json(path):
    return json.loads(open(path, "r").read())
