import json

import yaml


def write_mapping_to_json(mapping, path):
    json_string = json.dumps(mapping, indent=4)
    with open(path, "w") as sink:
        sink.write(json_string)


def read_mapping_from_json(path):
    return json.loads(open(path, "r").read())


def write_mapping_to_yaml(config, path):
    yaml.dump(config, open(path, "w"))


def read_mapping_from_yaml(path):
    return yaml.load(open(path, "r").read(), Loader=yaml.Loader)


def read_jsonl(path, ents_field=None, cats_field=None):
    data = []
    for line in open(path, "r"):
        text, entry = json.loads(line)
        if ents_field is not None:
            entry["ents"] = entry[ents_field]
        if cats_field is not None:
            entry["cats"] = entry[cats_field]

        data.append((text, entry))
    return data
