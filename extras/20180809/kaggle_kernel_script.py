import os
import numpy as np
import cv2
import pandas as pd
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
print('torch version:', torch.__version__)

import warnings
warnings.filterwarnings('ignore')


#######################################################################
# kaggle kernel
DATA_DIR  = '../input/severstal-steel-defect-detection'
USER_DATA = '../input/user-data'
SUBMISSION_CSV_FILE = 'submission.csv'


# overwrite to local path
DATA_DIR=\
    '/root/share/project/kaggle/2019/steel/data'
USER_DATA=\
    '/root/share/project/kaggle/2019/steel/code/dummy_01/__reference__/asimandia/user_data/dump'
SUBMISSION_CSV_FILE = \
    '/root/share/project/kaggle/2019/steel/code/dummy_01/__reference__/asimandia/user_data/dump/submission.csv'




IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]
CHECKPOINT_FILE = USER_DATA + '/model_unet_resnet18.pth'

# etc ############################################################
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError




#### model #####################################################################################
BatchNorm2d = nn.BatchNorm2d

class Basic(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, is_shortcut=False):
        super(Basic, self).__init__()
        self.conv1 = nn.Conv2d( in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1,      padding=1, bias=False)
        self.bn2   = BatchNorm2d(out_channel)

        self.is_shortcut =  in_channel != out_channel or stride!=1
        if self.is_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride, bias=False),
                BatchNorm2d(out_channel)
            )

    def forward(self, x):
        if self.is_shortcut:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        x = self.bn1(self.conv1(x))
        x = F.relu(x,inplace=True)
        x = self.bn2(self.conv2(x)) +  shortcut
        x = F.relu(x,inplace=True)
        return x

class Decode(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decode, self).__init__()
        self.top = nn.Sequential(
            nn.Conv2d( in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, e=None):

        #x = F.interpolate(x, size=(H,W), mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        if e is not None:
            x = torch.cat([x, e],1)
        x = self.top(x)
        return x

class Net(nn.Module):

    def __init__(self, in_channel=3, num_class=4):
        super(Net, self).__init__()

        self.encode = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
                BatchNorm2d(64),
                nn.ReLU(inplace=True),
                #nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )
        ])

        for in_channel, out_channel, stride, num_block in [
            [       64,          64,     1,       2],
            [       64,         128,     2,       2],
            [      128,         256,     2,       2],
            [      256,         512,     2,       2],
        ]:
            self.encode.append(
                nn.Sequential(
                   Basic( in_channel, out_channel,  stride=stride, ),
                *[ Basic(out_channel, out_channel,  stride=1,      ) for i in range(1, num_block) ]
                )
            )


        self.decode = nn.ModuleList([
            Decode( 512+256, 256),
            Decode( 256+128, 128),
            Decode( 128+ 64,  64),
            Decode(  64+ 64,  32),
            Decode(  32+  0,  16),
        ])

        self.logit = nn.Conv2d(16, num_class, kernel_size=1)



    def forward(self, x):
        batch_size,C,H,W = x.shape

        x = self.encode[0](x) ;  e0=x #; print('encode[0] :', x.shape)
        x = F.max_pool2d(x, kernel_size=3,stride=2,padding=1)

        x = self.encode[1](x) ;  e1=x #; print('encode[1] :', x.shape)
        x = self.encode[2](x) ;  e2=x #; print('encode[2] :', x.shape)
        x = self.encode[3](x) ;  e3=x #; print('encode[3] :', x.shape)
        x = self.encode[4](x) ;  e4=x #; print('encode[4] :', x.shape)

        #exit(0)
        x = self.decode[0](x,e3)      #; print('decode[0] :', x.shape)
        x = self.decode[1](x,e2)      #; print('decode[1] :', x.shape)
        x = self.decode[2](x,e1)      #; print('decode[2] :', x.shape)
        x = self.decode[3](x,e0)      #; print('decode[3] :', x.shape)
        x = self.decode[4](x)         #; print('decode[3] :', x.shape)

        #x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit

#### data #####################################################################################

def image_to_input(image):
    input = image.astype(np.float32)
    input = input[...,::-1]/255
    input = input.transpose(0,3,1,2)
    input[:,0] = (input[:,0]-IMAGE_RGB_MEAN[0])/IMAGE_RGB_STD[0]
    input[:,1] = (input[:,1]-IMAGE_RGB_MEAN[1])/IMAGE_RGB_STD[1]
    input[:,2] = (input[:,2]-IMAGE_RGB_MEAN[2])/IMAGE_RGB_STD[2]
    return input


class KaggleTestDataset(Dataset):
    def __init__(self):

        df =  pd.read_csv(DATA_DIR + '/sample_submission.csv')
        df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
        self.uid = df['ImageId'].unique().tolist()

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        return string

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, index):
        # print(index)
        image_id = self.uid[index]
        image = cv2.imread(DATA_DIR + '/test_images/%s'%(image_id), cv2.IMREAD_COLOR)
        return image, image_id


def null_collate(batch):
    batch_size = len(batch)

    input = []
    image_id = []
    for b in range(batch_size):
        input.append(batch[b][0])
        image_id.append(batch[b][1])

    input = np.stack(input)
    input = torch.from_numpy(image_to_input(input))
    return input, image_id


## -- kaggle --

def post_process(probability, threshold, min_size):

    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    predict = np.zeros((256, 1600), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predict[p] = 1
            num += 1

    return predict, num


def run_length_encode(mask):

    m = mask.T.flatten()
    if m.sum()==0:
        rle=''
    else:
        start  = np.where(m[1: ] > m[:-1])[0]+2
        end    = np.where(m[:-1] > m[1: ])[0]+2
        length = end-start

        rle = [start[0],length[0]]
        for i in range(1,len(length)):
            rle.extend([start[i],length[i]])

        rle = ' '.join([str(r) for r in rle])

    return rle


#########################################################################

def run_check_setup():

    if 0:
        logit_ref = np.load('/root/share/project/kaggle/2019/steel/code/dummy_01/__reference__/asimandia/user_data/dump/logits.npy')
        print(logit_ref[0,0,:5,:5],'\n')
        print(logit_ref[3,0,-5:,-5:],'\n')
        print(logit_ref.mean(),logit_ref.std(),logit_ref.max(),logit_ref.min(),'\n')
        exit(0)
    ##---------------------------------------------------------------------


    ## load net
    net = Net().cuda()
    net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage))

    ## load data
    image_id = ['004f40c73.jpg', '006f39c41.jpg', '00b7fb703.jpg', '00bbcd9af.jpg']
    image=[]
    for i in image_id:
        m = cv2.imread(DATA_DIR +'/test_images/%s'%i)
        image.append(m)
    image=np.stack(image)
    input = image_to_input(image)
    input = torch.from_numpy(input).cuda()

    #run here!
    net.eval()
    with torch.no_grad():
        logit = net(input)
        probability= torch.sigmoid(logit)

    print('input: ',input.shape)
    print('logit: ',logit.shape)
    print('')
    #---
    input = input.data.cpu().numpy()
    logit = logit.data.cpu().numpy()

    if 1:
        print(logit[0,0,:5,:5],'\n')
        # model.py: calling main function ...
        # [[-7.826587 -8.261375 -8.326101 -8.426055 -8.462243]
        #  [-8.661071 -9.520123 -9.517914 -9.558877 -9.531863]
        #  [-8.719453 -9.551636 -9.454987 -9.59043  -9.662381]
        #  [-8.832735 -9.584188 -9.600956 -9.693104 -9.741772]
        #  [-8.853442 -9.562226 -9.713482 -9.748312 -9.788805]]
        print(logit[3,0,-5:,-5:],'\n')
        # [[-9.883025  -9.795594  -9.624824  -9.273483  -8.205625 ]
        #  [-9.840044  -9.769805  -9.610718  -9.24963   -8.224977 ]
        #  [-9.62722   -9.583279  -9.434898  -9.1131315 -8.140734 ]
        #  [-9.213085  -9.2032385 -9.131121  -8.847996  -8.001736 ]
        #  [-8.9456    -8.9553175 -8.915957  -8.72905   -8.091164 ]]
        print(logit.mean(),logit.std(),logit.max(),logit.min(),'\n')
        #-9.166691 3.239937 2.0910003 -102.86145



def run_make_submission_csv():

    threshold = 0.5
    min_size  = 3500

    ## load net
    print('load net ...')
    net = Net().cuda()
    net.load_state_dict(torch.load(CHECKPOINT_FILE, map_location=lambda storage, loc: storage))
    print('')


    ## load data
    print('load data ...')
    dataset = KaggleTestDataset()
    print(dataset)
    loader  = DataLoader(
        dataset,
        sampler     = SequentialSampler(dataset),
        batch_size  = 8,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )



    ## start here
    image_id_class_id = []
    encoded_pixel = []


    net.eval()

    start = timer()
    for t,(input, image_id) in enumerate(loader):
        print('\r t = 3%d / 3%d  %s  %s : %s'%(
              t, len(loader)-1, str(input.shape), image_id[0], time_to_str((timer() - start),'sec'),
        ),end='', flush=True)

        input = input.cuda()
        with torch.no_grad():
            logit = net(input)
            probability= torch.sigmoid(logit)


        probability = probability.data.cpu().numpy()
        batch_size = len(image_id)
        for b in range(batch_size):
            p = probability[b]
            for c in range(4):
                predict, num = post_process(p[c], threshold, min_size)
                rle = run_length_encode(predict)

                image_id_class_id.append(image_id[b]+'_%d'%(c+1))
                encoded_pixel.append(rle)


    df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
    df.to_csv(SUBMISSION_CSV_FILE, index=False)


    print('')



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    #run_check_setup()
    run_make_submission_csv()

    print('\nsucess!')















