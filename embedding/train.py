import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import numpy as np 
from torch.optim import lr_scheduler
import shutil
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition RGB Test Case')

parser.add_argument('--desc', type=str, default='037_98.242188_91.176471_checkpoint.pth.tar', help='descripton of the channels of image')
parser.add_argument('--gpu_id',type = int,default=0, help='foo help')

parser.add_argument('--activation', type=str, default='Sigmoid', help='descripton of the channels of image')
parser.add_argument('--pooling', type=str, default='both', help='descripton of the channels of image')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--test', dest='test', action='store_true',help='evaluate model on test set')
#parser.add_argument('--print', dest='print', action='store_true',help='print prediction results')
parser.add_argument('--modelfile', type=str, default='model_best.path.tar', help='descripton of the channels of image')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(args.gpu_id)


class CovidFeatureDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, root):
        'Initialization'
        self.root = root
        feature_files = os.listdir(root) 
        #print(feature_files) 

        self.ids = [] 
        self.data = [] 
        self.labels = [] 

        classes = {'covid':0, 'non-covid':1} 

        for f in feature_files:
            #print(f) 

            if args.test: 
                f_splits = f.split('_') 
                label = 0 #no lable 
                s_id = f
            else: 
                f_splits = f.split('_') 
                label = classes[f_splits[2]] 
                s_id = f #int(f_splits[-1].split('.')[0]) 

            with open(os.path.join(root,f), 'rb') as pk_file:
                #features = pickle.load(pk_file)     
                [features,num_sets,slices_to_process] = pickle.load(pk_file)  
           
            #print(features.shape) 
            #apply L2 norm 
            #features = features/np.linalg.norm(features, ord=2, axis=1, keepdims=True)
            
            feature_max = np.amax(features,axis = 0) 
            feature_max = np.expand_dims(feature_max, axis=0) 
            feature_avg = np.mean(features,axis = 0) 
            feature_avg = np.expand_dims(feature_avg, axis=0) 

            feature_both = np.concatenate((feature_max,feature_avg),axis=1) 
            #print(feature_max.shape, feature_avg.shape,feature_both.shape) 
            self.data.append((feature_max,feature_avg,feature_both)) 
            self.ids.append(s_id) 
            self.labels.append(label) 

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.ids)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sid = self.ids[index]
        # Load data and get label
        X = self.data[index] 
        y = self.labels[index]

        return X,y,sid 


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
 '''

  def __init__(self,input_size=512,activation="ReLU"):
    super().__init__()

    if activation=="Sigmoid":
        act = nn.Sigmoid()
    elif  activation=="Tanh":
        act = nn.Tanh()
    else:
        act = nn.ReLU()

    self.layers = nn.Sequential(
      nn.Flatten(),
      nn.Linear(input_size, 256),
      act, # nn.Sigmoid(), #Sigmoid, ReLU or Tanh
      nn.Dropout(0.5), 
      nn.Linear(256, 32),
      act, #nn.Sigmoid(),
      nn.Dropout(0.5),
      nn.Linear(32, 2)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)
 

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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    torch.save(state, cur_path)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    if is_best:
        shutil.copyfile(cur_path, best_path)
 
  
def train(mlp,train_loader,feature_index,optimizer,loss_function): 

    # Set current loss value
    total_loss = 0 
    # Iterate over the DataLoader for training data
    count = 0
    correct = 0 
    # switch to train mode
    mlp.train()

    for i, data in enumerate(train_loader):      

      #print("i={}".format(i)) 

      # Get inputs
      inputs, targets, ids = data
      
      inputs = inputs[feature_index].cuda() 
      targets= targets.cuda()  

      print(inputs.shape) 
      print(targets.shape) 

      #print(ids.shape) 
      
      # Zero the gradients
      optimizer.zero_grad()
      
      # Perform forward pass
      outputs = mlp(inputs)

      #print(outputs.shape) 
      
      prec1 = accuracy(outputs.data, targets)

      #print(prec1.item) 

      correct += prec1.item()


      # Compute loss
      loss = loss_function(outputs, targets)
      total_loss += loss* outputs.size(0)
      count +=  outputs.size(0)

      #print(correct,count) 
      
      # Perform backward pass
      loss.backward()      
      # Perform optimization
      optimizer.step()
      
      #if (i+1) % 10 == 0:
      #    print('i = {} acc = {} loss = {}'.format(i, correct/count,total_loss/count))
      #input('dbg') 

    return correct, total_loss, count 

def validate(mlp,val_loader,feature_index,loss_function):

    mlp.eval()

    correct = 0 
    count = 0 
    total_loss = 0

    if args.evaluate: 
        if args.test:  
            fp = open("MLP-test-results.txt","w") 
        else:
            fp = open("MLP-validate-results.txt","w") 

    with torch.no_grad():
        for i, data in enumerate(val_loader):      
            # Get inputs
            inputs, targets, ids = data
            #print(i,targets,ids) 
            inputs = inputs[feature_index].cuda() 
            targets= targets.cuda()  
            # Perform forward pass
            outputs = mlp(inputs)
            prec1 = accuracy(outputs.data, targets)
            correct += prec1.item()
 
            # Compute loss
            loss = loss_function(outputs, targets)
            total_loss += loss* outputs.size(0)
            count +=  outputs.size(0)

            if args.evaluate: 
                _, pred = outputs.data.topk(1, 1, True, True)
                pred = pred.t()
                target2 = targets.view(1, -1).expand_as(pred)
                pred2 = pred.cpu().numpy()[0]
                target2 = target2.cpu().numpy()[0]
                output2 = outputs.data.cpu().numpy()
                for outind in range(len(ids)):
                    fp.write("{},{},{},{:4.3f},{:4.3f}\n".format(ids[outind],target2[outind],pred2[outind],output2[outind][0], output2[outind][1]) )

    if args.evaluate: 
        fp.close()

    return correct, total_loss, count 




if __name__ == '__main__':
  
  train_feature_dir = 'features_{}_train'.format(args.desc)
  if args.test: 
      val_feature_dir = 'features_{}_test'.format(args.desc)
  else: 
      val_feature_dir = 'features_{}_val'.format(args.desc)

  print("train_feature_dir = {}".format(train_feature_dir))    
  print("val_feature_dir = {}".format(val_feature_dir))    

  train_dataset = CovidFeatureDataset(root=train_feature_dir)
  val_dataset = CovidFeatureDataset(root=val_feature_dir)

  print("training data length = {}, validation data length = {}".format(len(train_dataset),len(val_dataset))) 

  pooling = args.pooling # "both" 
  activation = args.activation 

  print("pooling = {}, activation = {}".format(pooling, activation)) 


  saveLocation = "./checkpoint/"
  if not os.path.exists(saveLocation):
      os.makedirs(saveLocation) 

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
  val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

  
  # Initialize the MLP
  if pooling == "Max":
      mlp = MLP(input_size=512,activation=activation).cuda()
      feature_index = 0 
  elif pooling == "Avg":
      mlp = MLP(input_size=512,activation=activation).cuda()
      feature_index = 1
  else: 
      mlp = MLP(input_size=1024,activation=activation).cuda()
      feature_index = 2 
  
  # Define the loss function and optimizer
  loss_function = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
  scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)


  if args.evaluate:

      model_path = saveLocation + args.modelfile
      if not os.path.exists(model_path):
          model_path = saveLocation + "model_best.pth.tar" 
      print("Evaluation model_path = {}".format(model_path)) 

      params = torch.load(model_path)

      mlp.load_state_dict(params['state_dict'])
      mlp.cuda()
      mlp.eval() 

      correct, total_loss, count = validate(mlp,val_loader,feature_index,loss_function)

      prec1 = correct/count
      avg_loss = total_loss/count  
 
      print("Validation Prec@1 = {} loss = {}".format(prec1,avg_loss)) 

      exit() 



  best_prec1 = 0  
  best_loss = 100  
  is_best = False 

  # Run the training loop
  for epoch in range(0, 200): # 5 epochs at maximum
    correct, total_loss, count = train(mlp,train_loader,feature_index,optimizer,loss_function) 
    prec1_train = correct/count
    avg_loss = total_loss/count 
    print("Train Epoch =  {} Prec@1 = {} loss = {}".format(epoch, prec1_train,avg_loss)) 

    # evaluate on validation set
    if prec1_train > 0.85: 

        correct, total_loss, count = validate(mlp,val_loader,feature_index,loss_function)

        avg_loss = total_loss/count 
        scheduler.step(avg_loss)

        prec1 = correct/count
     
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1) 
        best_loss = min(best_loss,avg_loss)  
 
        print("Validation Epoch =  {} Prec@1 = {} loss = {}".format(epoch, prec1,avg_loss)) 

        checkpoint_name = "%s_%s_%s_%03d_%f_%f_%s" % (activation,pooling, args.desc, epoch + 1, prec1_train, prec1, "checkpoint.pth.tar")
        if is_best: # or (epoch + 1)%1==0:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': mlp.state_dict(),
                    'best_prec1': best_prec1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                     }, is_best, checkpoint_name, saveLocation)
  '''  
  checkpoint_name = "%s_%s_%s_%03d_%f_%f_%s" % (activation,pooling, args.desc, epoch + 1, prec1_train, prec1, "checkpoint.pth.tar")
  save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': mlp.state_dict(),
        'best_prec1': best_prec1,
        'best_loss': best_loss,
        'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint_name, saveLocation)

  ''' 
  # Process is complete.
  print('Training process has finished.')

