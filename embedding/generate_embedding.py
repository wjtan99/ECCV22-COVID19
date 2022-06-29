#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:52:03 2019

@author: esat
"""

import os, sys
import numpy as np

import time
import argparse

from ptflops import get_model_complexity_info

import torch,cv2
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm
import pickle,json

import argparse

datasetFolder="../BERT/datasets"

sys.path.insert(0, "../BERT/")
sys.path.insert(0,'.')
sys.path.insert(0,'../BERT/datasets/')


import models
from VideoSpatialPrediction3D_bert_embedding import VideoSpatialPrediction3D_bert
import video_transforms

from arcface import ArcFace 

from utils import RandomResampler, SymmetricalResampler, SymmetricalSequentialResampler


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="3"


model_names = sorted(name for name in models.__dict__
    if not name.startswith("__")
    and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__all__)

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--settings', metavar='DIR', default='../BERT/datasets/settings',
                    help='path to datset setting files')
parser.add_argument('--dataset', '-d', default='covid',
                    choices=["ucf101", "hmdb51", "smtV2", "window","videoreloc","covid"],
                    help='dataset: ucf101 | hmdb51 | smtV2')

parser.add_argument('--dataset_root', type=str, default='/media/ubuntu/MyHDataStor3/datasets/COV19D/',
                    help='dataset root directory')

parser.add_argument('--desc', type=str, default='rrr', help='descripton of the channels of image')


parser.add_argument('--subset', '-t', default='val',
                    choices=["train", "val","test"],
                    help='subset: train | val | test')


parser.add_argument('--arch', '-a', metavar='ARCH', default='rgb_r2plus1d_32f_34_bert10',
                    choices=model_names)

parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')

parser.add_argument('-refer_nums', default=10, type=int, metavar='Ref_N',
                    help='The number of refers in each category')

parser.add_argument('-w', '--window', default=3, type=int, metavar='V',
                    help='validation file index (default: 3)')


parser.add_argument('-v', '--val', dest='window_val', action='store_true',
                    help='Window Validation Selection')

parser.add_argument('--gpu_id',type = int,default=0, help='foo help')

parser.add_argument('--modelfile', type=str, default='model_best.pth.tar',help='model filename')


multiGPUTest = False
multiGPUTrain = True
ten_crop_enabled = False
num_seg = 32
num_seg_3D=1


debug = False 

result_dict = {}

def buildModel(model_path,num_categories):
    model=models.__dict__[args.arch](modelPath='', num_classes=num_categories,length=num_seg_3D)
    params = torch.load(model_path)

    if multiGPUTrain:
        new_dict = {k[7:]: v for k, v in params['state_dict'].items()} 
        model_dict=model.state_dict() 
        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
    elif multiGPUTest:
        model=torch.nn.DataParallel(model)
        new_dict={"module."+k: v for k, v in params['state_dict'].items()} 
        model.load_state_dict(new_dict)
        
    else:
        model.load_state_dict(params['state_dict'])
    model.cuda()
    model.eval() 
 
    #HEAD = ArcFace(in_features = 512, out_features = 2, device_id = [0])
    #HEAD.load_state_dict(params['state_dict_head'] )
    #HEAD.eval() 
    
    return model#,HEAD

def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0) 

    #print("batch_size = ",batch_size) 

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()

    target2 = target.view(1, -1).expand_as(pred)
    correct = pred.eq(target2)

    #print("pred = {}, target = {}, correct = {}".format(pred,target,correct)) 

    correct = correct.cpu().numpy() 
    #print("correct  = {}".format(correct)) 

    correct = np.sum(correct) 
    #print("correct  = {}".format(correct)) 
     
    return correct  


#include a global bbox for every scan, and mask for every slice 
def cv2_loader2(root,path,size,is_color=True,bbox=None):

    if is_color:
        flag = cv2.IMREAD_COLOR         # > 0
    else:
        flag = cv2.IMREAD_GRAYSCALE     # = 0
    interpolation = cv2.INTER_LINEAR

    #already read grayscale 
    flag = cv2.IMREAD_GRAYSCALE 
    img_path = os.path.join(root,path) 
    #print(img_path) 

    img = cv2.imread(img_path, flag)
    #print(img.shape) 

    if img is None:
       print("Could not load file %s" % (img_path))
       input("debugging not enough frame images") 
       sys.exit()
    '''
    if img.shape != (512,512): #this happens 
        img = cv2.resize(img,(512,512), interpolation) 
    ''' 
    
    mask_path = os.path.join(root,'mask', path+'_refined.jpg') 
    #print(mask_path) 
    mask = cv2.imread(mask_path, flag)
    if mask is None:
       print("Could not load file %s" % (mask_path))
       input("debugging not enough mask images") 
       sys.exit()
 
    #print(img.shape,mask.shape) 

    xmin,ymin,xmax,ymax = bbox  

    #print(xmin,ymin,xmax,ymax)  
    
    #print(np.max(mask),np.min(mask))   
    
    img_mask = img.copy() 
    black_ind = mask ==0 
    img_mask[black_ind] = 0 

    '''
    img_crop = img[ymin:ymax,xmin:xmax] 
    img_mask_crop = img_mask[ymin:ymax,xmin:xmax] 

    #print(img_crop.shape,img_mask_crop.shape) 


    #img_merged = cv2.merge([img,img_crop,img_mask_crop])
    onlyonce = False 
    if onlyonce: 
        cv2.imwrite("debug/img.jpg",img) 
        cv2.imwrite("debug/img_crop.jpg",img_crop) 
        cv2.imwrite("debug/img_mask.jpg",img_mask) 
        cv2.imwrite("debug/img_mask_crop.jpg",img_mask_crop) 
        onlyonce = False 

    img_crop = cv2.resize(img_crop, size, interpolation)
    img_mask_crop = cv2.resize(img_mask_crop, size, interpolation)
    img = cv2.resize(img, size, interpolation)
         

    img = cv2.resize(img, size, interpolation)
    #mask = cv2.resize(mask, size, interpolation)
    img_mask = cv2.resize(img_mask, size, interpolation)
    
    img_mask = np.expand_dims(img_mask, axis=2)
    img      = np.expand_dims(img, axis=2)

    img_merged = np.concatenate((img,img_mask), axis=2)
    print(img_merged.shape) 
    ''' 

    #img_merged = cv2.merge([img,mask,img_mask])
    img_merged = cv2.merge([img,img,img])
 
    #img_merged = cv2.merge([img_mask,img_mask,img_mask])
 
    #06/23 evening 11:17 added 
    #img_merged = img_merged[ymin:ymax,xmin:xmax,:] 

    img_merged = cv2.resize(img_merged, size, interpolation)

    #img_merged = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) # cv2.COLOR_BGR2RGB)

    return img_merged 


import torch.utils.data as data

class SingleVideoTest(data.Dataset):

    def __init__(self,
                 root, 
                 video_path,
                 width,
                 height,
                 length,
                 video_transform=None,
                ):

        debug = False 
        
        self.root = root
        self.video_path = video_path 
        
        self.width = width
        self.height = height
        self.video_transform = video_transform
        
        video_path_full = os.path.join(root,video_path)
        if debug:
            print(video_path_full)
        
        img_files = os.listdir(video_path_full)
        img_files.sort(key=lambda x: int(x.split('_')[-3][3:]) )
        if debug:
            print(img_files)

        num_imgs = len(img_files)                
        num_segs = num_imgs//length
     
        
        last_seg_len = num_imgs - num_segs*length 
        
        if last_seg_len >0:
            num_segs += 1 
            
        if debug:
            print(num_imgs,num_segs,last_seg_len)

            
        self.data = [] 
        
        for s in range(num_segs): 
            if debug:
                print("segment = ", s)
            
            slices = [] 
            if s<num_segs-1: 
                for k in range(s*length,(s+1)*length):
                    slices.append(img_files[k])
            else:
                for k in range(max(num_imgs-length,0),num_imgs):
                    slices.append(img_files[k])                
            if debug:
                print(slices) 
            
            if len(slices)<length:
                slices = SymmetricalResampler.resample(slices, length)
            
            self.data.append(slices)
        
        if debug:
            print("data_len = ", len(self.data))
        

    def __getitem__(self, index):

        #print("Starting index {} data generator".format(index))  
        slices = self.data[index]
        imageList = []                 
        for s in slices: 
            img_path  = os.path.join(self.video_path,s)

            img = cv2_loader2(self.root, img_path,  (width, height))
            #img = pil_loader(self.root,img_path)



            imageList.append(img) 
            
        #print("number of images = ", len(imageList))    
        img_out_segs = []         
        for i, img in enumerate(imageList): 
            img_out = self.video_transform(img)
            #print(i, img_out.shape)
            img_out_segs.append(img_out)                         
            
        clip_input = torch.stack(img_out_segs)            
        
        #print(clip_input.shape)

        return clip_input, slices  

    def __len__(self):
        return len(self.data)


debug = False 

def main():
    global args
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_id)

    if '64f' in args.arch:
        length=64
    elif '32f' in args.arch:
        length=32
    elif '8f' in args.arch:
        length=8    
    else:
        length=16

    print("arch = {}, length = {} ".format(args.arch, length)) 

    modelLocation="./checkpoint/"+args.dataset+"_"+args.arch+"_split"+str(args.split)  #  '_2021_01_07_18_27_43'
    model_path = os.path.join('../BERT/',modelLocation,args.modelfile)
    print("model_path = {}".format(model_path)) 

    if not os.path.exists(model_path):
        #model_path = os.path.join(modelLocation,'203_96.849593_90.702479_checkpoint.pth.tar') #'model_best.pth.tar') 
        model_path = os.path.join(modelLocation,'model_best.pth.tar') 

        print("model_path = {}".format(model_path)) 
        input("check model path") 


    dataset = args.dataset_root 
    print("dataset root directory = {}".format(dataset))  

    val_setting_file = "%s_rgb_split%d.txt" % (args.subset, args.split)
    val_split_file = os.path.join(args.settings, args.dataset, val_setting_file)
    print("val_split_file = {}".format(val_split_file)) 

    if not os.path.exists(val_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % (args.settings))

    
    start_frame = 0
    if args.dataset=='ucf101':
        num_categories = 101
    elif args.dataset=='hmdb51':
        num_categories = 51
    elif args.dataset=='smtV2':
        num_categories = 174
    elif args.dataset=='window':
        num_categories = 3
    elif args.dataset=='videoreloc':
        num_categories = 160
    elif args.dataset=='covid':
        num_categories = 2

    model_start_time = time.time()
    BERT  = buildModel(model_path,num_categories)
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition model is loaded in %4.4f seconds." % (model_time))
    
    # flops, params = get_model_complexity_info(spatial_net, (3,length, 112, 112), as_strings=True, print_per_layer_stat=False)
    # #flops, params = get_model_complexity_info(spatial_net, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    f_val = open(val_split_file , "r")
    val_list = f_val.readlines()
    print("we got %d test videos" % len(val_list))
    
    scale = 1.0
    input_size = int(224 * scale)
    width = height = int(input_size*1.25)  # 1.25    
    print("scale= {}, input_size = {}, width = {}, height = {}".format(scale,input_size,width,height)) 
  
    clip_mean = [0.43216, 0.394666, 0.37645]  
    clip_std = [0.22803, 0.22145, 0.216989]  

    normalize = video_transforms.Normalize(mean=clip_mean,
                                std=clip_std)

    val_transform = video_transforms.Compose([
                video_transforms.CenterCrop((input_size)),
                video_transforms.ToTensor(),
                normalize,
            ])

    dims = (height, width,3)    


    id_of_class = []
    val_data_len = len(val_list)

    for line in val_list:
        id_of_class.append(line.split()[1])

    num_of_class = len(set(id_of_class))

    print("id_of_class = {}".format(set(id_of_class)) ) 
    print("num_of_class = {}".format(num_of_class)) 

    #result_list = []


    classes = {"covid":0,"non-covid":1} 
    
    if True:
        
        Predictions = dict() 

        count = 0    
        for line in tqdm(val_list):
            # line_info = line.split(" ")
            # clip_path = os.path.join(data_dir,line['id'])
            # duration = int(line_info[1])
            # input_video_label = int(line_info[2]) 

            target = classes[line.split()[1]]    
            if debug:          
                print("line = {}".format(line)) 

            output_file = args.dataset_root + 'features/{}.npy'.format(line.split(' ')[0])
            output_path = os.path.dirname(output_file) 
            if not os.path.exists(output_path):
               os.makedirs(output_path)


            fn_splits = line.split()[0].split('/')  
            fn = "{}_{}_{}".format(fn_splits[0], fn_splits[1],fn_splits[2]) 

            img_dir = line.split()[0] 
            mask_dir = 'mask/' + img_dir 

            img_path = os.path.join(dataset, img_dir)
            mask_path = os.path.join(dataset, mask_dir)
 
            first_slice = int(line.split()[2]) 
            last_slice = int(line.split()[3]) 

            #print(img_dir,mask_dir,img_path,first_slice,last_slice) 

            slices_all = [] 
            for s in range(first_slice,last_slice+1):
                slices_all.append("{}.jpg".format(s))     
   
            if debug:
                print("slices_all = ", slices_all) 

            #slices_to_process, num_sets = SymmetricalSequentialResampler(slices_all,length)

            slices_to_process = [] 
            img_files = slices_all 

            num_imgs = len(img_files)                
            num_segs = num_imgs//length            
            last_seg_len = num_imgs - num_segs*length 
        
            if last_seg_len >0:
                num_segs += 1 
        
            for s in range(num_segs): 
                if debug and False:
                    print("segment = ", s)
            
                slices = [] 
                if s<num_segs-1: 
                    for k in range(s*length,(s+1)*length):
                        slices.append(img_files[k])
                else:
                    for k in range(max(num_imgs-length,0),num_imgs):
                        slices.append(img_files[k])                
                if debug and False:
                    print(slices) 
            
                if len(slices)<length:
                    slices = SymmetricalResampler.resample(slices, length)
                slices_to_process.append(slices) 

            if debug and False:
                print(slices_to_process) 
             

            features = [] 
            preds = [] 

            for ind in range(len(slices_to_process)): 
           
                slices = slices_to_process[ind] 
                if debug:
                    print(ind,slices) 

                imageList = [] 
                
                for s in slices: 
                    img_file  = os.path.join(img_path,s) 
                    mask_file = os.path.join(mask_path,s +'_refined.jpg') 
                    #print(img_file,mask_file) 

                    img  = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE) 
                    mask = cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE)                     

                    img_mask = img.copy() 
                    black_ind = mask ==0 
                    img_mask[black_ind] = 0 
                     
                    img_merged = cv2.merge([img,mask,img_mask])
                    img_merged_resized = cv2.resize(img_merged, (height,width), cv2.INTER_LINEAR)
                    #img_mask = cv2.resize(img_mask, (height,width), cv2.INTER_LINEAR)
                    #img_merged = cv2.cvtColor(img_mask,cv2.COLOR_GRAY2RGB) # cv2                     
                    imageList.append(img_merged_resized) #CenterCrop 


                rgb_list=[] 
                for ind in range(len(imageList)):   
                    cur_img = imageList[ind].copy() 
                    cur_img_tensor = val_transform(cur_img)
                    rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
                       
                input_data=np.concatenate(rgb_list,axis=0) 
                #print("input_data.shape = {}".format(input_data)) 
  
                with torch.no_grad():
                    imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
                    #print("imgDataTensor.shape = {}".format(imgDataTensor)) 

                    imgDataTensor = imgDataTensor.view(-1,length,3,input_size,input_size).transpose(1,2)

                    output, input_vectors, sequenceOut, maskSample, embedding = BERT(imgDataTensor)
                    #04/30 added ArcFace head 
                    #output = HEAD(embedding, targets)

                    _, pred = output.topk(1, 1, True, True)
                    pred = pred.cpu().numpy()[0][0]
                    output2 = output.data.cpu().numpy()[0]
                    preds.append((pred,output2)) 
                  
                    embedding = embedding.cpu().numpy().squeeze(axis=0)  
                    #print(embedding.shape) 

                    features.append(embedding) 

                if debug: 
                    print("segment_index = {}, clip_len = {}".format(ind, length)) 
                    print("features len = {}".format(len(features)) )
                    input("debug feature") 
 
            features_final = np.asarray(features) # .cpu().numpy            
            if debug:
                print(features_final.shape) 
            #input("dbg") 
            Predictions[fn] = preds

            np.save(output_file, features_final)  
                

            count +=1 

    
        fp = open("Embedding-{}-results.txt".format(args.subset),'w') 
        for key in Predictions:
            preds = Predictions[key] 
            if "non-covid" in key:
                true = 1
            else:
                true = 0 
            fp.write("{},{}".format(key,true))
            for p in preds:
                fp.write(",{},{:4.3f},{:4.3f}".format(p[0],p[1][0],p[1][1]))
            fp.write("\n")
        fp.close()  

if __name__ == "__main__":
    main()

