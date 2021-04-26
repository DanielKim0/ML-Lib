import os
import shutil
import pickle
import numpy as np

def compress_files(path, data=None, arrays=None):
    if not os.path.exists("temp"):
        os.mkdir("temp")
    else:
        raise OSError("temp directory exists. Please delete before continuing.")

    try:
        if arrays:
            for key in arrays:
                np.save(os.path.join("temp", key + ".npy"), arrays[key].numpy())

        if data:
            with open(os.path.join("temp", "meta.pickle"), "wb") as f:
                f.write(pickle.dumps(data))

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
        return data, arrays
    finally:
        shutil.rmtree("temp")
