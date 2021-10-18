# Formation-Jetson-IA

##Pré-requis :
* 1 carte Jetson Nano avec carte SD et alimentation avec écran, clavier et souris
* 1 PC Windows, Mac ou Linux
* 1 adaptateur Micro-SD

# Flash de la jetson Nano
* Télécharger l'image de la carte SD officielle de NVIDIA : https://developer.nvidia.com/embedded/jetpack#install
* Flasher la carte SD préparée avec Etcher

## Préparation de la carte SD sur Windows : 
* Télécharger et installer SD Memory Card Formatter : https://www.sdcard.org/downloads/formatter_4/eula_windows/
* Sélectionner la carte SD
* Select "Quick format"
* Laisser "Volume label" vide
* Cliquer sur "Format"

## Préparation de la carte SD sur Mac ou Linux :
* Télécharget et installer Etcher : https://www.balena.io/etcher
* Sélectionner l'image zippée
* Insérer la carte SD et ignorer l'éventuel message d'erreur

# Premier démarrage de la Jetson Nano
* Insérer la carte SD dans le slot de la Jetson
* Connecter l'écran, un clavier et une souris
* Alimenter la Jetson avec un cable Micro-USB ou une alimentation externe (jumper à modifier sur la carte pour sélectionner l'alimentation)


# Configuration de la jetson Nano pour le Comuter Vision et le Deep Learning
* Utiliser les capacités d'alimentation maximale : `sudo nvpmodel -m 0` puis `sudo jetson_clocks`
* Faisons de la place : `sudo apt-get purge libreoffice*`, `sudo apt-get clean`
* Update du système : `sudo apt update && sudo apt upgrade`

## Install Tensorflow
https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

## Dépendances et pré-requis
```bash
sudo apt-get install git cmake
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libhdf5-serial-dev hdf5-tools
sudo apt-get install python3-dev
sudo apt-get install nano locate
sudo apt-get install libfreetype6-dev python3-setuptools
sudo apt-get install protobuf-compiler libprotobuf-dev openssl
sudo apt-get install libssl-dev libcurl4-openssl-dev
sudo apt-get install cython3
sudo apt-get install libxml2-dev libxslt1-dev
```

## Mise à jour de CMake
```bash
wget http://www.cmake.org/files/v3.13/cmake-3.13.0.tar.gz
tar xpvf cmake-3.13.0.tar.gz cmake-3.13.0/
cd cmake-3.13.0/
./bootstrap --system-curl
make -j4
```

## Mise à jour du profil bash :
```bash
echo 'export PATH=/home/nvidia/cmake-3.13.0/bin/:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## Préparer l'installation d'OpenCV
```bash
sudo apt-get install build-essential pkg-config
sudo apt-get install libtbb2 libtbb-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libxvidcore-dev libavresample-dev
sudo apt-get install libtiff-dev libjpeg-dev libpng-dev
sudo apt-get install python-tk libgtk-3-dev
sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
sudo apt-get install libv4l-dev libdc1394-22-dev
```

## Création d'un environnement virtuel pour Python
```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
rm get-pip.py
sudo pip install virtualenv virtualenvwrapper
nano ~/.bashrc :
* export WORKON_HOME=$HOME/.virtualenvs
* export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
* source /usr/local/bin/virtualenvwrapper.sh
source ~/.bashrc
```
Redémmarrer le terminal puis :
```bash
mkvirtualenv py3 -p python3
workon py3cv4
```

## Installer le compilateur Protobuf (pour que Tensorflow aille vite, 1h d'installation)
```bash
wget https://raw.githubusercontent.com/jkjung-avt/jetson_nano/master/install_protobuf-3.6.1.sh
sudo chmod +x install_protobuf-3.6.1.sh
./install_protobuf-3.6.1.sh
```

## Installer le compilateur Protobuf dans l'environnement virtuel :
```bash
workon py3
cd ~
cp -r ~/src/protobuf-3.6.1/python/ .
cd python
python setup.py install --cpp_implementation
```

## Installation de Tensorflow
```bash
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 'tensorflow<2'
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow where JP_VERSION The major and minor version of JetPack you are using, such as 42 for JetPack 4.2.2 or 33 for JetPack 3.3.1.
```
```bash
sudo pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION tensorflow==$TF_VERSION+nv$NV_VERSION

Where:
JP_VERSION
The major and minor version of JetPack you are using, such as 42 for JetPack 4.2.2 or 33 for JetPack 3.3.1.
TF_VERSION
The released version of TensorFlow, for example, 1.13.1.
NV_VERSION
The monthly NVIDIA container version of TensorFlow, for example, 19.01.
```

## Step #12: Install the TensorFlow Object Detection API on Jetson Nano
```bash
cd ~
workon py3cv4
git clone https://github.com/tensorflow/models
cd ~
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py install
```

## Installation et compilation d'OpenCV sur Jetson Nano
```bash
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.1.2 opencv
mv opencv_contrib-4.1.2 opencv_contrib
workon py3cv4
cd opencv
mkdir build
cd build
```

Compiler OpenCV avec CMake :
```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D WITH_CUDA=ON \
	-D CUDA_ARCH_PTX="" \
	-D CUDA_ARCH_BIN="5.3,6.2,7.2" \
	-D WITH_CUBLAS=ON \
	-D WITH_LIBV4L=ON \
	-D BUILD_opencv_python3=ON \
	-D BUILD_opencv_python2=OFF \
	-D BUILD_opencv_java=OFF \
	-D WITH_GSTREAMER=ON \
	-D WITH_GTK=ON \
	-D BUILD_TESTS=OFF \
	-D BUILD_PERF_TESTS=OFF \
	-D BUILD_EXAMPLES=OFF \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/home/`whoami`/opencv_contrib/modules ..
```

La suite prend approximativement 2,5 heures
```bash 
make -j4
sudo make install
cd ~/.virtualenvs/py3/lib/python3.6/site-packages/
ln -s /usr/local/lib/python3.6/site-packages/cv2/python3.6/cv2.cpython-36m-aarch64-linux-gnu.so cv2.so
```

## Autres dépendances
```bash
pip install matplotlib scikit-learn
pip install pillow imutils scikit-image
pip install dlib
pip install lxml progressbar2
```