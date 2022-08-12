from mrcnn.utils import Dataset
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray


class TablesDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):
        # define class
        self.add_class("dataset", 1, "column")
        self.add_class("dataset", 2, "row")

        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
        # find all images
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4]

            # skip all images after 100 if we are building the train set
            if is_train and int(image_id[4:8]) <= 580:
                continue
            # skip all images before 100 if we are building the test/val set
            if not is_train and int(image_id[4:8]) > 580:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            coors, class_name = boxes[i]
            row_s, row_e = coors[1], coors[3]
            col_s, col_e = coors[0], coors[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(class_name))
        return masks, asarray(class_ids, dtype='int32')

    # extract bounding boxes from an annotation file

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


# class that defines and loads the TableStruct dataset
def extract_boxes(filename):
    # load and parse the file
    tree = ElementTree.parse(filename)
    # get the root of the document
    root = tree.getroot()
    # extract each bounding box
    boxes = list()
    for obj in root.findall('.//object'):
        box = obj.find('bndbox')
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = [xmin, ymin, xmax, ymax]

        class_name = obj.find('name').text
        boxes.append((coors, class_name))

    # extract image dimensions
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)

    return boxes, width, height
