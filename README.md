# ECCV22-COVID19

Implementation of "Two Stage COVID19 Classification Using BERT Features" for ECCV-2022 MIA COV19D Competition. 

There are Four parts in this project

## Preprocess
Preprocess the CT-scan volume images: check the image size, extract bounding box and percentage of the the lung in the whole image, select images for 3D CNN

## Segmentation
A UNet segmentation network is trained. It is used to segment lung mask of an image. 

## BERT
A 3D CNN network with BERT for CT-scan volume classification and embedding feature extraction 

## embedding
Generate embeddings in the first stage 3D-CNN-BERT network. These embeddings are used as input in the 2nd stage BERT classification.   

# License
The code of 3D-CNN-BERT-COVID19 is released under the MIT License. There is no limitation for both academic and commercial usage.
