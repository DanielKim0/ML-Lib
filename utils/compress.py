import os
import shutil
import pickle
import numpy as np

def save_layers(layers):
    os.mkdir(os.path.join("temp", "layers"))
    for i in range(len(layers)):
        if layers[i].weighted:
            data = self.prep_save(layer)
            pickle.dump(os.path.join("temp", "layers", f"{str(i)}_layer.pickle"), data)
            np.save(os.path.join("temp", "layers", f"{str(i)}_w.pickle"), data[0].numpy())
            np.save(os.path.join("temp", "layers", f"{str(i)}_b.pickle"), data[1].numpy())
            layer.finish_save(data)

def load_layers(path):
    layers = []
    count = 0

    while os.path.exists(os.path.join(path, f"{str(count)}_layer.pickle")):
        layer = pickle.load(os.path.join(path, f"{str(count)}_layer.pickle"))
        if os.path.exists(os.path.join(path, f"{str(count)}_w.pickle")):
            layer.w = np.load(os.path.join(path, f"{str(count)}_w.pickle"))
            layer.b = os.path.join(path, f"{str(count)}_b.pickle")
        layers.append(layer)
        count += 1
    return layers

def compress_files(path, data=None, arrays=None, layers=None):
    if not os.path.exists("temp"):
        os.mkdir("temp")
    else:
        raise OSError("temp directory exists. Please delete before continuing.")

    try:
        if arrays:
            for key in arrays:
                if isinstance(arrays[key], list):
                    item = [i.numpy() for i in arrays[key]]
                    np.save(os.path.join("temp", key + ".npy"), item)
                else:
                    np.save(os.path.join("temp", key + ".npy"), arrays[key].numpy())

        if data:
            pickle.dump(os.path.join("temp", "meta.pickle"), data)

        if layers:
            save_layers(layers)

        shutil.make_archive(path, "zip", "temp")
        shutil.move(path + ".zip", path)
    finally:
        shutil.rmtree("temp")
    
def uncompress_files(path):
    if not os.path.exists("temp"):
        os.mkdir("temp")
    else:
        raise OSError("temp directory exists. Please delete before continuing.")

    try:
        shutil.unpack_archive(path, "temp", "zip")
        
        data = None
        if os.path.exists(os.path.join("temp", "meta.pickle")):
            data = pickle.load(open(os.path.join("temp", "meta.pickle"), "rb"))

        arrays = dict()
        for _, _, files in os.walk("temp"):
            for name in files:
                if name != "meta.pickle":
                    key = os.path.splitext(os.path.basename(name))[0]
                    arrays[key] = np.load(os.path.join("temp", name))

        if os.path.exists(os.path.join("temp", "layers")):
            layers = load_layers(os.path.join("temp", "layers"))
            data["layers"] = layers
        
        return data, arrays
    finally:
        shutil.rmtree("temp")
