Generate 3D-CNN-BERT embeddings 
===
 
### Generate embedding data 

python generate_embedding.py --split=5 --subset=train --desc=addSeq_affine --gpu_id=2 --modelfile=047_97.330729_90.909091_checkpoint.pth.tar

Features will be generated and saved in directory featurs_subset_desc. 


### Dependency - Use same enviroment as the BERT 
Recreate the Pytorch-1.7 Anaconda container enviroment by running conda install --name myenv --file pytorch-1.7.txt


