import random
import monai ##monai: medical open network for AI
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation


## ========= divide into train, val, test ========= ##
# defining train,val, test set splitting function
def partitioning_loading(imageFiles_labels,opt):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    targets = opt['task']['targets']

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['subjectkey']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        images.append(image)
        labels.append(label)

    transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(opt['data_augmentation']['resize']),
                              ToTensor()])

    # number of total / train,val, test
    num_total = len(images)
    
    # training data are split into training_warmup and training_learning
    num_warmup = int(num_total*opt['data_split']['warmup_size'])
    num_train1 = int((num_total - num_warmup)/2)
    num_train2 = num_total - num_warmup - num_train1

    # image and label information of train
    images_warmup = images[:num_warmup]
    labels_warmup = labels[:num_warmup]
    images_train1 = images[num_warmup:num_warmup + num_train1]
    labels_train1 = labels[num_warmup:num_warmup + num_train1]
    images_train2 = images[num_warmup + num_train1:]
    labels_train2 = images[num_warmup + num_train1:]

    train_warmup_set = ImageDataset(image_files=images_warmup,labels=labels_warmup,transform=transform)
    train_set1 = ImageDataset(image_files=images_train1,labels=labels_train1,transform=transform)
    train_set2 = ImageDataset(image_files=images_train2,labels=labels_train2,transform=transform)

    partition = {}
    partition['train_warmup'] = train_warmup_set 
    partition['train1'] = train_set1
    partition['train2'] = train_set2

    return partition
## ====================================== ##

def loading(imageFiles_labels,opt):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    targets = opt['task']['targets']

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['subjectkey']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        images.append(image)
        labels.append(label)

    transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(opt['data_augmentation']['resize']),
                              ToTensor()])

    data_set = ImageDataset(image_files=images,labels=labels,transform=transform)
    
    return data_set
## ====================================== ##