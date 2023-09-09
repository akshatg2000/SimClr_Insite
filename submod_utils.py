import numpy as np
import h5py
import os
from PIL import Image
import matplotlib.pyplot as plt
from submodlib import FacilityLocationFunction, DisparitySumFunction, DisparityMinFunction, LogDeterminantFunction
import shutil
import time
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle
import random
import os
import enum
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from spear.labeling import PreLabels, LFSet
import numpy as np
from spear.cage import Cage
from spear.labeling import labeling_function, ABSTAIN, preprocessor, continuous_scorer
import re
torch.manual_seed(7)
random.seed(7)


class Img2Vec():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
    RESNET_OUTPUT_SIZES = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
    }

    def __init__(self, cuda=False, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        print("Using: ", model, " for feature extraction.")
        self.device = torch.device("cuda" if cuda else "cpu")

        self.model_name = model
        self.layer_output_size = layer_output_size

        self.model, self.extraction_layer = self._get_model_and_layer(
            model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        if type(img) == list:
            a = [self.normalize(self.to_tensor(self.scaler(im))) for im in img]
            images = torch.stack(a).to(self.device)
            if self.model_name in ['alexnet', 'vgg']:
                my_embedding = torch.zeros(len(img), self.layer_output_size)
            elif self.model_name == 'densenet':
                my_embedding = torch.zeros(len(img), self.layer_output_size, 7, 7)
            else:
                my_embedding = torch.zeros(len(img), self.layer_output_size, 1, 1)

            def copy_data(m, i, o):
                my_embedding.copy_(o.data)

            h = self.extraction_layer.register_forward_hook(copy_data)
            with torch.no_grad():
                h_x = self.model(images)
            h.remove()

            if self.model_name in ['alexnet', 'vgg']:
                return my_embedding.numpy()[:, :]
            elif self.model_name == 'densenet':
                return torch.mean(my_embedding, (2, 3), True).numpy()[:, :, 0, 0]
            else:
                return my_embedding.numpy()[:, :, 0, 0]
        else:
          image = self.normalize(self.to_tensor(
              self.scaler(img))).unsqueeze(0).to(self.device)

          if self.model_name in ['alexnet', 'vgg']:
              my_embedding = torch.zeros(1, self.layer_output_size)
          elif self.model_name == 'densenet':
              my_embedding = torch.zeros(1, self.layer_output_size, 7, 7)
          else:
              my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

          def copy_data(m, i, o):
              my_embedding.copy_(o.data)

          h = self.extraction_layer.register_forward_hook(copy_data)
          with torch.no_grad():
              h_x = self.model(image)
          h.remove()

          if self.model_name in ['alexnet', 'vgg']:
              return my_embedding.numpy()[0, :]
          elif self.model_name == 'densenet':
              return torch.mean(my_embedding, (2, 3), True).numpy()[0, :, 0, 0]
          else:
              return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name.startswith('resnet') and not model_name.startswith('resnet-'):
            model = getattr(models, model_name)(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = self.RESNET_OUTPUT_SIZES[model_name]
            else:
                layer = model._modules.get(layer)
            return model, layer
        elif model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'vgg':
            # VGG-11
            model = models.vgg11_bn(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = model.classifier[-1].in_features # should be 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'densenet':
            # Densenet-121
            model = models.densenet121(pretrained=True)
            if layer == 'default':
                layer = model.features[-1]
                self.layer_output_size = model.classifier.in_features # should be 1024
            else:
                raise KeyError('Un support %s for layer parameters' % model_name)

            return model, layer
        elif model_name == 'googlenet':
            model = models.googlenet(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 1024
            else:
                layer = model._modules.get(layer)

            return model, layer
        else:
            raise KeyError('Model %s was not found' % model_name)

# return list of subsets of images sorted by class
def classwise_subsets(path="/home/akshit/Desktop/CKIM/data/", numcls=5):
    import numpy as np
    from sklearn.model_selection import train_test_split

    data = np.load(path+'retinamnist.npz')
    subsets = []

    for i in range(numcls):
        idx = np.where(data['train_labels'] == i)[0]
        img = data["train_images"][idx]
        subsets.append(img)

    return np.array(subsets)

# return seed set of limited number of images per class
def balanced_seedset(cls_size=100,path="/home/akshit/Desktop/CKIM/data/"):
    import numpy as np
    
    img = classwise_subsets(path=path)
    selected_images = [arr[:cls_size] if cls_size<len(arr) else arr for arr in img]
    seed_set = np.concatenate(selected_images, axis=0)
    return seed_set


def extractFeaturesWithLabels(data, dir="/home/akshit/Desktop/CKIM/data/aptos/", method='resnet50'):
  hdf5_path = os.path.join(dir, 'features_labels_'+method+'.hdf5')
  f = h5py.File(hdf5_path, mode='w')
  img2vec = Img2Vec(cuda=False, model=method)
  images = data['train_images'].astype('uint8')
  image_list = []
  labels = data['train_labels']
  imgs = []
  index = 0
  for image in images:
    # print(np.shape(image))
    img = Image.fromarray(image).convert('RGB')
    imgs.append(img)
    image_list.append(index)
    index += 1
  
  vectors = img2vec.get_vec(imgs)

  print("Features from ", method, vectors.shape)
  f.create_dataset("features", data=vectors)
  f.create_dataset("images", data=image_list)
  f.create_dataset("labels", data=labels)
  f.close()

def subset_selection(dir="/home/akshit/Desktop/CKIM/data/aptos/",model = "resnet50",budget = 40,algo = 'LogDeterminantFunction'):
    
    features_file = "features_labels_"+model+".hdf5"
    hf = h5py.File(os.path.join(dir, features_file), 'r')
    data = hf["features"][:]
    num_images = len(data)
    features_list = {"resnet18":512, "resnet50":2048, "densenet":1024, "vgg":4096}
    num_features = features_list[model]
    
    print(f'Number of images to process - {num_images}')
    print(f'Starting selection using {algo}')

    if algo == 'FacilityLocationFunction':
        objFL = FacilityLocationFunction(n=num_images, data=np.array(data), separate_rep=False,
                                        mode="dense", metric="euclidean")
        greedyList = objFL.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False,
                                    verbose=False)
    elif algo == 'DisparitySumFunction':
        objFL = DisparitySumFunction(n=num_images, data=np.array(data), mode="dense", metric="euclidean")
        greedyList = objFL.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False,
                                    verbose=False)
    elif algo == 'DisparityMinFunction':
        objFL = DisparityMinFunction(n=num_images, data=np.array(data), mode="dense", metric="euclidean")
        greedyList = objFL.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False,
                                    verbose=False)
    elif algo == 'LogDeterminantFunction':
        lambda_value = 1
        objFL = LogDeterminantFunction(n=num_images, data=np.array(data), mode="dense", metric="euclidean",
                                        lambdaVal=lambda_value)
        greedyList = objFL.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False,
                                    verbose=False)

    
    selected_idx_1 = [greedyList[i][0] for i in range(len(greedyList))]
    label_array = np.array(hf["labels"])
    selected_labels_1 = label_array[selected_idx_1]
    sorted_indices = np.argsort(selected_idx_1)
    selected_idx_1 = np.array(selected_idx_1).astype(int)

    selected_idx = selected_idx_1[sorted_indices]
    selected_labels = selected_labels_1[sorted_indices]

    unique_elements, counts = np.unique(selected_labels, return_counts=True)

    print(f"#######  {algo}  #########")
    for element, count in zip(unique_elements, counts):
        print(f"Class {element} has count {count}")
    
    return selected_idx,selected_labels

def random_subset_selection(dir="/home/akshit/Desktop/CKIM/data/aptos/",budget = 40, model='resnet50'):
    
    features_file = "features_labels_"+model+".hdf5"
    hf = h5py.File(os.path.join(dir, features_file), 'r')
    data = hf["labels"][:]
    selected_idx = sorted(random.sample(range(len(data)), budget))
    selected_labels = hf["labels"][selected_idx]

    return selected_idx,selected_labels

def lf_creator(idx, label, model="resnet50",dir="/home/akshit/Desktop/CKIM/data/aptos/"):
    
    @continuous_scorer()
    def similarity_scorer(x,**kwargs):
        import numpy as np
        from scipy.spatial.distance import cosine

        # change this 
        features_file = "features_labels_"+model+".hdf5"
        hf = h5py.File(os.path.join(dir, features_file), 'r')
        data = hf["features"][:]
        exemplar = data[idx]
        similarity = 1 - cosine(exemplar.flatten(), x.flatten())
        return similarity
    
    @labeling_function(name=f"LF_{idx}",cont_scorer=similarity_scorer, label=label)
    def LF_similarity_scorer(x, **kwargs):
        return label
        
    return LF_similarity_scorer

def createLFs(selected_idx, selected_labels, model="resnet50",dir = "/home/akshit/Desktop/CKIM/data/aptos/"):
    
    # Declaring Class Labels
    
    enum_dict = {"NORMAL": 0, "DISEASED": 1}
    ClassLabels = enum.Enum('ClassLabels', enum_dict)
    LFS = []

    labels_list = []
    for i in selected_labels:
        labels_list.append(ClassLabels.NORMAL) if i==0 else None
        labels_list.append(ClassLabels.DISEASED) if i==1 else None

    # Creating Labelling Functions
    for i in range(len(selected_idx)):
        LFS.append(lf_creator(selected_idx[i], labels_list[i],model=model,dir=dir))

    return LFS, ClassLabels

def generate_pkl(selected_idx, selected_labels, model="resnet50", algo = 'FacilityLocationFunction', dir = "/home/akshit/Desktop/CKIM/data/aptos/",path = "/home/akshit/Desktop/CKIM/cage/aptos/"):
    from spear.labeling import PreLabels
    import numpy as np
    
    LFS, ClassLabels = createLFs(selected_idx, selected_labels, model=model, dir=dir)
    rules = LFSet("BM_LF")
    rules.add_lf_list(LFS)

    features_file = "features_labels_"+model+".hdf5"
    hf = h5py.File(os.path.join(dir, features_file), 'r')
    xu = hf["features"][:]
    xl = hf["features"][selected_idx]
    yl = hf["labels"][selected_idx]

    # Labelled Set
    l_noisy_labels = PreLabels(name="retina_l",
                                data=xl,
                                gold_labels=yl,
                                rules=rules,
                                labels_enum=ClassLabels,
                                num_classes=2)
    l_noisy_labels.generate_pickle(f'{path}labelled_{model}_{algo}_{len(selected_labels)}.pkl')
    l_noisy_labels.generate_json(f'{path}labels_{model}_{algo}_{len(selected_labels)}.json')
    

    #Lake Set
    u_noisy_labels = PreLabels(name="retina_ul",
                               data=xu,
                               rules=rules,
                               labels_enum=ClassLabels,
                               num_classes=2)
    u_noisy_labels.generate_pickle(f'{path}unlabelled_{model}_{algo}_{len(selected_labels)}.pkl')
    
    del l_noisy_labels, u_noisy_labels, LFS, rules, hf
    
def get_cage_labels(model = 'resnet50', algo = 'FacilityLocationFunction', qt = 0.9, qc = 0.85, metric_avg = ['macro'], n_epochs = 100, lr = 0.01, n_lfs=40, path = "/home/akshit/Desktop/CKIM/cage/aptos/"):
    from spear.cage import Cage
    # n_lfs = budget
    cage = Cage(path_json = f'{path}labels_{model}_{algo}_{n_lfs}.json', n_lfs = n_lfs)
    probs = cage.fit_and_predict_proba(path_pkl = f'{path}unlabelled_{model}_{algo}_{n_lfs}.pkl', path_test = f'{path}labelled_{model}_{algo}_{n_lfs}.pkl', path_log = f'{path}log.txt', qt = qt, qc = qc, metric_avg = metric_avg, n_epochs = n_epochs, lr = lr)
    labels = np.argmax(probs, 1)
    del cage, probs
    return labels

def get_accuracy(labels,selected_idx,data):
    from sklearn.metrics import accuracy_score
    
    yu = data["train_labels"][:]
    #removing gold-labelled indices 
    yu = [element for i, element in enumerate(yu) if i not in selected_idx]
    labels = [element for i, element in enumerate(labels) if i not in selected_idx]

    return accuracy_score(labels, yu)


