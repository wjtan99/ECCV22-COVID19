#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm


# In[2]:


covidx_dir = '/media/ubuntu/MyHDataStor2/datasets/COVID-19/ICCV-MIA/'
covidx_img_dir= covidx_dir 
covidx_mask_dir= covidx_dir + 'mask/' 
data_list_dir = '/media/ubuntu/MyHDataStor2/products/COVID-19/ICCV-MAI/3D-CNN-BERT/BERT/datasets/settings/covid/'

print(covidx_dir)
print(covidx_img_dir)
print(covidx_mask_dir)
print(data_list_dir)


# In[3]:


def load_labels_covidx(label_file):
    """Loads image filenames, classes, and bounding boxes"""
    fnames, classes, bboxes, ratios = [], [], [], []
    
    fp = open(label_file, 'r')    
    lines = fp.readlines() 
    fp.close()
    lines = [x.strip() for x in lines]
    
    for line in lines:
        fname, cls, xmin, ymin, xmax, ymax, ratio = line.split()
        fnames.append(fname)
        classes.append(cls)
        bboxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        ratios.append(float(ratio))            
    
    return fnames, classes, bboxes,ratios,lines


# In[ ]:


subsets = ['val']
split = 4 

areas = dict() 
slice_lens = dict() 

for subset in subsets:     

    areas[subset] = [] 
    slice_lens[subset] = [] 
    
    
    label_file = covidx_dir+'{}_ICCV_MAI.txt'.format(subset)
    fnames, classes, bboxes,ratios,lines = load_labels_covidx(label_file)

    list_file =  data_list_dir + '{}_rgb_split{}.txt'.format(subset,split) 
    print(list_file) 
    
    fp = open(list_file,'w')  
    
    count = {'covid':0,'non-covid':0}
    
    #train/covid/ct_scan_0/0.jpg covid 0 18 512 344 0.004721
   
    covid_scan_ids = [] 
    non_covid_scan_ids = [] 
    for f in fnames: 
        if 'non-covid' in f: 
            non_covid_scan_ids.append(f.split('/')[2])
        else:
            covid_scan_ids.append(f.split('/')[2])
            
    covid_scan_ids = set(covid_scan_ids)    
    covid_scan_ids = list(covid_scan_ids)
    non_covid_scan_ids = set(non_covid_scan_ids)    
    non_covid_scan_ids = list(non_covid_scan_ids)
    
    covid_scan_ids.sort(key = lambda x: int(x.split('_')[-1]))
    non_covid_scan_ids.sort(key = lambda x: int(x.split('_')[-1]))
    
    #print("covid_scan_ids = {}".format(covid_scan_ids))
    #print("non_covid_scan_ids = {}".format(non_covid_scan_ids))
    
    all_scan_ids = {"covid": covid_scan_ids, "non-covid": non_covid_scan_ids}
    
    for c in all_scan_ids: 
        
        pbar = tqdm(total=len(all_scan_ids[c]))
        
        for s in all_scan_ids[c]:
            pbar.update()
            #print("class = {}, scan_id = {}".format(c,s))
            
            s_files = [x for x in lines if x.split()[1]==c and x.split()[0].split('/')[2] == s]
            s_files.sort(key = lambda x: int(x.split()[0].split('/')[-1].split('.')[0]) )
            
            s_ratios = [float(x.split()[-1]) for x in s_files]    
            
            #print(s_files)
            
            
            #print(len(s_ratios))
            
            s_ratio_max = np.max(s_ratios) 
            #print(s_ratio_max)        
          
            for thresh_ind in range(7,0,-1):
                #print(thresh_ind)
                thresh = thresh_ind/10 
                s_ind = np.where(s_ratios >= s_ratio_max*thresh)[0]  
                
                if len(s_ind)>=2:
                    if s_ind[-1]-s_ind[0] >=2: 
                        break 
            

            xmins  = [float(x.split()[2]) for x in s_files[s_ind[0]:s_ind[-1]+1]]    
            ymins  = [float(x.split()[3]) for x in s_files[s_ind[0]:s_ind[-1]+1]]    
            xmaxs  = [float(x.split()[4]) for x in s_files[s_ind[0]:s_ind[-1]+1]]    
            ymaxs  = [float(x.split()[5]) for x in s_files[s_ind[0]:s_ind[-1]+1]]              
            
            xmin = int(np.min(xmins))
            ymin = int(np.min(ymins))
            xmax = int(np.max(xmaxs))   
            ymax = int(np.max(ymaxs))

            slice_lens[subset].append(s_ind[-1]-s_ind[0])    
            areas[subset].append((xmax-xmin)*(ymax-ymin)/512/512)    
            
            #print(xmin,ymin,xmax,ymax) 
            
            #print(s_ind) 
            
            s_dir = "{}/{}/{}".format(subset,c,s) 
            line = "{} {} {} {} {} {} {} {}\n".format(s_dir,c,s_ind[0],s_ind[-1],xmin,ymin,xmax,ymax)
            #print(line)            
            
            #if (s_ind[-1]-s_ind[0]+1) < 16:
            #if len(s_ind)<16: 
            #    continue 
                
            fp.write(line)     
            
            '''
                        
            means,stds = [], []     
            means1,stds1 = [], []     
            means2,stds2 = [], []     
                
            
            count = 0 
            for ind in range(s_ind[0],s_ind[-1]+1): 
                
                f = s_files[ind]
                #print(f)
                 
                img_fn = covidx_img_dir + f.split()[0]
                mask_fn = covidx_mask_dir + f.split()[0]
                
                #print(img_fn)
                #print(mask_fn)
                
                img = cv2.imread(img_fn,0)
                if img.shape != (512,512):
                    img = cv2.resize(img,(512,512))
                
                mask = cv2.imread(mask_fn,0)
                
                #print(img.shape,mask.shape)
                
                black_ind = mask==0                 
                img_mask = img.copy()
                img_mask[black_ind] = 0 
               
                
                img_crop = img[ymin:ymax,xmin:xmax] 
                mask_crop = mask[ymin:ymax,xmin:xmax] 
                img_mask_crop = img_mask[ymin:ymax,xmin:xmax] 
                
                means.append(np.mean(img/255))
                stds.append(np.std(img/255))
                
                means1.append(np.mean(img_crop/255))
                stds1.append(np.std(img_crop)/255)
                
                means2.append(np.mean(img_mask_crop/255))
                stds2.append(np.std(img_mask_crop/255))
                
                count += 1 
                
                    
                if count>50 and count<100 and False:
                #if True: 
                    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
                    ax1.imshow(img,cmap='gray')
                    ax2.imshow(img_crop,cmap='gray')
                    ax3.imshow(mask,cmap='gray')
                    ax4.imshow(img_mask,cmap='gray')               
                    ax5.imshow(img_mask_crop,cmap='gray')   
                    

            #input('dbg')            
            
            print(np.mean(means))
            print(np.mean(stds))
            print(np.mean(means1))
            print(np.mean(stds1))
            print(np.mean(means2))
            print(np.mean(stds2))
            
            break 
            '''
                        
        #break 
            
            
    fp.close() 
    
    
exit() 



#06/24 added for test dataset 
subsets = ['test']

#this code is to generate bounding box and percent of lung mask for test dataset
from tqdm import tqdm
debug = False 

c = 'covid' #all unknown for test dataset 

for subset in subsets: 
    
    slice_lens = dict() 
    
    label_file = covidx_dir+'{}_ICCV_MAI.txt'.format(subset)
    fnames, classes, bboxes,ratios,lines = load_labels_covidx(label_file)

    list_file =  data_list_dir + '{}_rgb_split{}.txt'.format(subset,split) 
    print(list_file) 
    
    fp = open(list_file,'w')  
  
    
    #use the following code on test datset 
    testsubsets = ["subset{}".format(x) for x in range(1,9)]
    print(testsubsets) 
    
    annots = dict()
    
    for testset in testsubsets:     

        print(testset)
        
        slice_lens[testset] = []
        
        annots[testset] = [] 
        
        subset_dir = 'test/' + testset     
        
        scan_dirs = os.listdir(covidx_img_dir + subset_dir)
        
        print(subset_dir)    
        
        
        pbar = tqdm(total=len(scan_dirs))
        
        for s in scan_dirs:
            
            pbar.update() 
            
            s_dir = subset_dir  + '/' + s         
            #print(s_dir)     
            
            s_files = [x for x in lines if testset in x and x.split()[0].split('/')[2] == s]           
            
            s_files.sort(key = lambda x: int(x.split()[0].split('/')[-1].split('.')[0]) )
            
            s_ratios = [float(x.split()[-1]) for x in s_files]   
            
            #print(s_files)
            #print(ratios)
            
            s_ratio_max = np.max(s_ratios) 
            #print(s_ratio_max)        
            

            for thresh_ind in range(7,0,-1):
                #print(thresh_ind)
                thresh = thresh_ind/10 
                s_ind = np.where(s_ratios >= s_ratio_max*thresh)[0]  
                
                if len(s_ind)>=2:
                    if s_ind[-1]-s_ind[0] >=0: 
                        break 
            
            slice_lens[testset].append(s_ind[-1]-s_ind[0])    
                

            xmins  = [float(x.split()[2]) for x in s_files[s_ind[0]:s_ind[-1]+1]]    
            ymins  = [float(x.split()[3]) for x in s_files[s_ind[0]:s_ind[-1]+1]]    
            xmaxs  = [float(x.split()[4]) for x in s_files[s_ind[0]:s_ind[-1]+1]]    
            ymaxs  = [float(x.split()[5]) for x in s_files[s_ind[0]:s_ind[-1]+1]]              
            
            xmin = int(np.min(xmins))
            ymin = int(np.min(ymins))
            xmax = int(np.max(xmaxs))   
            ymax = int(np.max(ymaxs))
            
            #print(xmin,ymin,xmax,ymax)             
            #print(s_ind) 
            
            s_dir = "{}/{}/{}".format(subset,testset,s)             
            line = "{} {} {} {} {} {} {} {}\n".format(s_dir,c,s_ind[0],s_ind[-1],xmin,ymin,xmax,ymax)
            #print(line)            
            fp.write(line)     
            #input('dbg')
        
        print(len(slice_lens[testset]),max(slice_lens[testset]),min(slice_lens[testset]))
        
    fp.close()


# In[ ]:




