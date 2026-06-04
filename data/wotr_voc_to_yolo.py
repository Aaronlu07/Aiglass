import os
import sys
import shutil
import xml.etree.ElementTree as ET


def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_image(jpeg_dir, image_id, xml_filename):
    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    for ext in exts:
        p = os.path.join(jpeg_dir, image_id + ext)
        if os.path.exists(p):
            return p
    if xml_filename:
        for ext in exts:
            if xml_filename.lower().endswith(ext):
                p = os.path.join(jpeg_dir, xml_filename)
                if os.path.exists(p):
                    return p
    return None


def parse_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.findtext("width", default="0"))
    h = int(size.findtext("height", default="0"))
    filename = root.findtext("filename", default="")
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", default="0"))
        ymin = float(bnd.findtext("ymin", default="0"))
        xmax = float(bnd.findtext("xmax", default="0"))
        ymax = float(bnd.findtext("ymax", default="0"))
        objects.append((name, xmin, ymin, xmax, ymax))
    return w, h, filename, objects


def to_yolo_box(w, h, xmin, ymin, xmax, ymax):
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(float(w), xmax)
    ymax = min(float(h), ymax)
    bw = max(0.0, xmax - xmin)
    bh = max(0.0, ymax - ymin)
    if w <= 0 or h <= 0 or bw <= 1 or bh <= 1:
        return None
    cx = (xmin + xmax) * 0.5 / float(w)
    cy = (ymin + ymax) * 0.5 / float(h)
    bw = bw / float(w)
    bh = bh / float(h)
    return cx, cy, bw, bh


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


def convert(root_dir):
    root_dir = os.path.abspath(root_dir)
    ann_dir = os.path.join(root_dir, "Annotations")
    img_dir = os.path.join(root_dir, "JPEGImages")
    split_dir = os.path.join(root_dir, "ImageSets", "Main")
    yolo_dir = os.path.join(root_dir, "yolo")
    labels_dir = os.path.join(yolo_dir, "labels")
    images_dir = os.path.join(yolo_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    rename = {
        "tactile_paving": "blind_road",
        "peristander": "person"
    }
    class_list = []
    class_index = {}

    def get_cls(name):
        name = rename.get(name, name)
        if name not in class_index:
            class_index[name] = len(class_list)
            class_list.append(name)
        return class_index[name], name

    for split in ["train", "val", "test"]:
        ids_path = os.path.join(split_dir, f"{split}.txt")
        if not os.path.exists(ids_path):
            continue
        ids = read_ids(ids_path)
        split_labels_dir = os.path.join(labels_dir, split)
        split_images_dir = os.path.join(images_dir, split)
        os.makedirs(split_labels_dir, exist_ok=True)
        os.makedirs(split_images_dir, exist_ok=True)
        for image_id in ids:
            xml_path = os.path.join(ann_dir, f"{image_id}.xml")
            if not os.path.exists(xml_path):
                continue
            w, h, xml_filename, objects = parse_voc(xml_path)
            img_path = find_image(img_dir, image_id, xml_filename)
            if not img_path:
                continue
            ext = os.path.splitext(img_path)[1].lower()
            dst_img_path = os.path.join(split_images_dir, f"{image_id}{ext}")
            link_or_copy(img_path, dst_img_path)
            label_path = os.path.join(split_labels_dir, f"{image_id}.txt")
            lines = []
            for name, xmin, ymin, xmax, ymax in objects:
                cls_id, _ = get_cls(name)
                yolo_box = to_yolo_box(w, h, xmin, ymin, xmax, ymax)
                if not yolo_box:
                    continue
                cx, cy, bw, bh = yolo_box
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            with open(label_path, "w", encoding="utf-8") as lf:
                lf.write("\n".join(lines))

    yaml_path = os.path.join(root_dir, "wotr_yolo.yaml")
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(f"path: {root_dir}\n")
        yf.write("train: yolo/images/train\n")
        yf.write("val: yolo/images/val\n")
        yf.write("test: yolo/images/test\n")
        yf.write(f"nc: {len(class_list)}\n")
        yf.write("names:\n")
        for name in class_list:
            yf.write(f"  - {name}\n")
    return yaml_path, class_list


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "WOTR")
    yaml_path, classes = convert(root)
    print("YOLO yaml:", yaml_path)
    print("Classes:", classes)
