import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import os

dataset = sklearn.datasets.fetch_olivetti_faces(data_home="./scikit_learn_data")
X_images = dataset.data.reshape(len(dataset.data), -1)
y_target = dataset.target

seed = 242010 #(M24MA2010)

train_indexes, val_indexes, test_indexes = [], [], []


rng = np.random.default_rng(seed)
for i in range(40):
    index = rng.permutation(np.where(y_target == i)[0])
    if len(index) < 8:
        raise ValueError(f"Not enough samples for class {i} to split into train, val, and test sets.")
    train_index = index[:6]
    val_index = index[6:8]
    test_index = index[8:]

    train_indexes.extend(train_index)
    val_indexes.extend(val_index)
    test_indexes.extend(test_index)     

X_train, X_val, X_test = X_images[train_indexes], X_images[val_indexes], X_images[test_indexes]
y_train, y_val, y_test = y_target[train_indexes], y_target[val_indexes], y_target[test_indexes]


test_org_shape = X_test.reshape(len(X_test), 64, 64).copy()
test_org_shape[:, 24:40, 24:40] = 0
test_masked = test_org_shape.reshape(len(X_test), -1)

os.makedirs("data", exist_ok=True)
np.savez("data/split_indices.npz", train=train_indexes, val=val_indexes, test=test_indexes, seed=seed)

with open("data/split_indices.csv", "w") as f:
    f.write("dataset_index,identity,split\n")
    for split_name, indexes in (("train", train_indexes), ("val", val_indexes), ("test", test_indexes)):
        for i in indexes:
            f.write(f"{i},{y_target[i]},{split_name}\n")

if __name__ == "__main__":
    print(f"Train set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    print(f"Test Robust: {test_masked.shape}")

    




