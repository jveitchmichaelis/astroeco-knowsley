import os

def read_names():
    out = {}
    names = read_all("obj.names")

    for i, n in enumerate(names):
        out[i] = n

    return out

def get_bucket_uri(image, bucket="gs://astroeco_data/knowsley_model_data/", subfolder=""):
    base = os.path.basename(image)
    return os.path.join(bucket, subfolder, base)

def read_all(file):
    with open(file, "r") as f:
        return [l.strip() for l in f.readlines()]

def load_labels(image, names):
    label_file = os.path.splitext(image)[0] + ".txt"
    labels = []

    for label in read_all(label_file):
        class_id, x, y, w, h = label.split(" ")
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        labels.append([names[int(class_id)], x-w/2, y-h/2, x+w/2, y+h/2])

    return labels

def generate_gcp_labels(file, mode="UNASSIGNED", subfolder=""):

    images = [f.strip() for f in open(file, "r").readlines()]
    names = read_names()

    for image in images:
        labels = load_labels(image, names)
        uri = get_bucket_uri(image, subfolder=subfolder)
        for label in labels:
            print("{},{},{},{},{},,,{},{},,".format(mode,uri,
						label[0],
						label[1],
						label[2],
						label[3],
						label[4]))

generate_gcp_labels("train.txt", mode="TRAIN", subfolder="train")
generate_gcp_labels("test.txt", mode="TEST", subfolder="test")
generate_gcp_labels("val.txt", mode="VAL", subfolder="val")
