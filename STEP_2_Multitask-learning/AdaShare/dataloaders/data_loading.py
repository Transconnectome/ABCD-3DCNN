import random
import monai ##monai: medical open network for AI
from monai.data import CSVSaver, ImageDataset, DistributedWeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, Flip, ToTensor
from monai.utils import set_determinism
from monai.apps import CrossValidation


## ========= divide into train, val, test ========= ##
def case_control_count(labels, dataset_type):
    if args.cat_target:
        for cat_target in args.cat_target:
            target_labels = []

            for label in labels:
                target_labels.append(label[cat_target])
            
            n_control = target_labels.count(0)
            n_case = target_labels.count(1)
            print('In {} dataset, {} contains {} CASE and {} CONTROL'.format(dataset_type, cat_target,n_case, n_control))
                     
# defining train,val, test set splitting function
def partitioning_loading(imageFiles_labels,opt):
    """This function used only for warming up phase"""
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    tasks = opt['task']['targets']

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['subjectkey']
        label = {}

        for label_name in tasks[:len(tasks)]:
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


    # image and label information of train
    images_warmup = images[:num_warmup]
    labels_warmup = labels[:num_warmup]
    images_train1 = images[num_warmup:num_warmup + num_train1]
    labels_train1 = labels[num_warmup:num_warmup + num_train1]
    images_train2 = images[num_warmup + num_train1:]
    labels_train2 = labels[num_warmup + num_train1:]
    #images_train = images[num_warmup:]
    #labels_train = labels[num_warmup:]

    train_warmup_set = ImageDataset(image_files=images_warmup,labels=labels_warmup,transform=transform)
    train_set1 = ImageDataset(image_files=images_train1,labels=labels_train1,transform=transform)
    train_set2 = ImageDataset(image_files=images_train2,labels=labels_train2,transform=transform)
    #train_set = ImageDataset(image_files=images_train,labels=labels_train,transform=transform)

    partition = {}
    partition['train_warmup'] = train_warmup_set 
    partition['train1'] = train_set1
    partition['train2'] = train_set2
    #partition['train1'] = train_set
    #partition['train2'] = train_set
    
    #counting case and control
    case_control_count(labels_warmup, 'warmup')
    case_control_count(labels_train1, 'train1')
    case_control_count(labels_train2, 'train2')

    return partition
## ====================================== ##

def loading(imageFiles_labels,opt):
    random.shuffle(imageFiles_labels)

    images = []
    labels = []
    tasks = opt['task']['targets']

    for imageFile_label in imageFiles_labels:
        image = imageFile_label['subjectkey']
        label = {}

        for label_name in tasks[:len(tasks)]:
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

