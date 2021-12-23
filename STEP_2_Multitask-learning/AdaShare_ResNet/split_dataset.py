import random
import monai ##monai: medical open network for AI
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation


## ========= divide into train, val, test ========= ##
# defining train,val, test set splitting function
def partition(imageFiles_labels,args):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    targets = args.cat_target + args.num_target

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['subjectkey']
        label = {}

        for label_name in targets[:len(targets)]:
            label[label_name]=imageFile_label[label_name]

        images.append(image)
        labels.append(label)


    resize = args.resize
    train_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    val_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    test_transform = Compose([ScaleIntensity(),
                               AddChannel(),
                               Resize(resize),
                              ToTensor()])

    # number of total / train,val, test
    num_total = len(images)

    # training data are split into training_warmup and training_learning
    num_train = int(num_total*(1 - args.warmup_size - args.val_size - args.test_size))
    num_warmup = int(num_train*args.warmup_size)
    num_val = int(num_total*args.val_size)
    num_test = int(num_total*args.test_size)

    # image and label information of train
    images_warmup = images[:num_warmup]
    labels_warmup = labels[:num_warmup]
    images_train = images[num_warmup:num_train]
    labels_train = labels[num_warmup:num_train]

    # image and label information of valid
    images_val = images[num_train:num_train+num_val]
    labels_val = labels[num_train:num_train+num_val]

    # image and label information of test
    images_test = images[num_total-num_test:]
    labels_test = labels[num_total-num_test:]

    train_warmup_set = ImageDataset(image_files=images_warmup,labels=labels_warmup,transform=train_transform)
    train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=train_transform)
    val_set = ImageDataset(image_files=images_val,labels=labels_val,transform=val_transform)
    test_set = ImageDataset(image_files=images_test,labels=labels_test,transform=test_transform)

    partition = {}
    partition['train_warmup'] = train_warmup_set 
    partition['train'] = train_set
    partition['val'] = val_set
    partition['test'] = test_set

    return partition
## ====================================== ##