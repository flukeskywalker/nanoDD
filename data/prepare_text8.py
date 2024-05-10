import os
import pathlib
import pickle
import requests
import zipfile

import numpy as np


def prepare_text8(data_dir: pathlib.Path):
    data_dir.mkdir(parents=True, exist_ok=False)
    data_url = "http://mattmahoney.net/dc/text8.zip"

    # download, extract and cleanup
    with open(data_dir / "text8.zip", "wb") as f:
        print(f"Downloading text8 to {data_dir}")
        f.write(requests.get(data_url).content)
        print("Done")
    with zipfile.ZipFile(data_dir / "text8.zip") as f:
        f.extractall(data_dir)
    os.remove(data_dir / "text8.zip")
    data = (data_dir / "text8").read_text()

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    # encode both to integers
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9): int(n * 0.95)]
    test_data = data[int(n * 0.95):]
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    test_ids = encode(test_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
    print(f"test has {len(test_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    np.save(data_dir / "train.npy", train_ids)
    np.save(data_dir / "val.npy", val_ids)
    np.save(data_dir / "test.npy", test_ids)
    print(f"Saved to {data_dir / 'train.npy'}, {data_dir / 'val.npy'}, {data_dir / 'test.npy'}")

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(data_dir / "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"text8 dataset downloaded and prepared in dir {data_dir}")


if __name__ == "__main__":
    prepare_text8(pathlib.Path(__file__).resolve().parent.parent / "text8")
