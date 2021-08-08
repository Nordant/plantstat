import numpy as np
import pandas as pd
import random, os, cv2, glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from google_drive_downloader import GoogleDriveDownloader as gdd


def set_seed(seed=0):
    '''
    Set seed for data reproducibility.
    Args:
        seed - int argument (default - 0)
    '''
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('--- Seed: "%i" ---' %seed)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    '''
    Class that returns image tensors, fake labels 
    and image paths for the dataloader.
    '''
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def dataloader(path, batch_size):
    '''
    Make PyTorch DataLoader object for images in the custom directory.
    Args:
        path - path to the directory (str)
        batch_size - the size of batches (int)
    '''
    num_images = len(glob.glob(path+'/*/*'))
    if batch_size > num_images:
        raise ValueError('Batch size greater than the number of images')
    else:
        transformer = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
        dataset = ImageFolderWithPaths(path, transformer)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                shuffle=False)
        return dataloader
              

class OpenStomataPredictor:
    '''
    The main class for stomata open/close classes prediction.
    '''
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size
        self.classes = ['close', 'open']

    def predict(self, save=False, f_format='excel'):
        '''
        Main function for stomata open/close classification.
        Args:
            save - save prediction in a local directory or not
            f_format - format of data saving (if save = True): 'csv' or 'excel' (default)
        '''
        assert f_format in {'excel', 'csv'}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_loader = dataloader(self.path, self.batch_size)
        
        gdd.download_file_from_google_drive(file_id='1qNpC2fhEtzeZAS3zXk1bx5RlQOwTkkEB',
                                            dest_path='./stomata_model_resnet50.pkl')
        print('\n')
        model = torch.load('stomata_model_resnet50.pkl', map_location=torch.device(device))
        model.eval()
        
        test_img_paths = []
        test_preds = []
        
        for inputs, _, paths in tqdm(data_loader, position = 0, leave = True):
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                preds = model(inputs)
            test_preds.append(torch.sigmoid(preds)[:, 1].data.cpu().numpy())
            test_img_paths.extend(paths)
        test_preds = np.concatenate(test_preds)
        
        self.test_img_paths_ = test_img_paths
        self.test_preds_ = test_preds
        self.test_classes_ = [self.classes[int(round(i))] for i in self.test_preds_]
        
        self.report_ = pd.DataFrame({'image': pd.Series(self.test_img_paths_).apply(lambda x: x.split('\\')[-1]),
                                     'probability': self.test_preds_,
                                     'class': self.test_classes_,
                                     'path': self.test_img_paths_})
        
        if save == True and f_format == 'csv':
            self.report_.to_csv('OpenStomata_report.csv')
        elif save == True and f_format == 'excel':
            self.report_.to_excel('OpenStomata_report.xlsx', sheet_name = 'preds')
        else:
            pass
        
        OpenStomata = self.test_preds_.round().sum() / len(self.test_preds_) * 100
        print('\nDone!')
        print('Open: {:.1f}%'.format(OpenStomata))
        print('Close: {:.1f}%'.format(100 - OpenStomata))

    def visualize(self, n_imgs=16, save=False):
        '''
        Visualize some random stomata open/close classification results.
        Args:
            n_imgs - images to visualize (int, default = 16)
            save = whether to save the output plot in local directory or not
        '''
        if n_imgs > len(self.test_preds_):
            raise ValueError('n_imgs arg greater than the number of images')
        else:
            plt.figure(figsize=(15, 15))
            for idx, i in enumerate(random.sample(range(len(self.test_preds_)), n_imgs)):
                path = self.test_img_paths_[i]
                plt.subplot(int(np.ceil(n_imgs / 4)), 4, idx+1)
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                plt.imshow(image)
                plt.axis('off')
                plt.title('Predicted class: {} ({:.4f})'.format(self.classes[int(round(self.test_preds_[i]))], self.test_preds_[i]))
            
            if save == True:
                plt.savefig('OpenStomata_report.png', dpi = 200)
            plt.show()
