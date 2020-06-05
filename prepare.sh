#!/bin/bash

# download and extract VOC 2007 trainval split
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar -P data/
mkdir data/VOCtrainval_06-Nov-2007
tar xf data/VOCtrainval_06-Nov-2007.tar --directory data/VOCtrainval_06-Nov-2007/

# download and extract VOC 2007 test split
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar -P data/
mkdir data/VOCtest_06-Nov-2007
tar xf data/VOCtest_06-Nov-2007.tar --directory data/VOCtest_06-Nov-2007/

# download and extract edgeboxes proposals
wget https://groups.inf.ed.ac.uk/hbilen-data/data/WSDDN/EdgeBoxesVOC2007test.mat -P data/
wget https://groups.inf.ed.ac.uk/hbilen-data/data/WSDDN/EdgeBoxesVOC2007trainval.mat -P data/

# download pretrained alexnet weights
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth -P states/

# download pretrained VGG weights
wget https://download.pytorch.org/models/vgg16-397923af.pth -P states/

# download pretrained alexnet based wsddn weights
wget https://www.dropbox.com/s/lifv1ywa98a2p4y/alexnet_epoch_20.pt -P states/

# build the docker image
docker build . -t wsddn.pytorch
