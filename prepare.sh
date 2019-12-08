# download and extract VOC 2007 trainval split
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar -P data/
mkdir data/VOCtrainval_06-Nov-2007
tar xf data/VOCtrainval_06-Nov-2007.tar --directory data/VOCtrainval_06-Nov-2007/

# download and extract VOC 2007 test split
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar -P data/
mkdir data/VOCtest_06-Nov-2007
tar xf data/VOCtest_06-Nov-2007.tar --directory data/VOCtest_06-Nov-2007/

# download and extract selective search windows boxes
wget http://www.cs.cmu.edu/~spurushw/hw2_files/selective_search_data.tar -P data/
tar xf data/selective_search_data.tar --directory data/

# download pretrained alexnet weights
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth -P states/

# download pretrained wsddn weights
wget https://www.dropbox.com/s/rpti37b6afsnb62/pretrained.zip -P states/
unzip states/pretrained.zip -d states/.

# build the docker image
docker build . -t wsddn.pytorch
