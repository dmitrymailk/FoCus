import json
import pickle


def save_object_to_json(obj, file_path):
    with open(file_path, "w") as f:
        json.dump(obj, f)


def save_object_to_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
