# Formation Jetson IA

## Pré-requis :
* 1 carte Jetson Nano avec carte SD et alimentation avec écran, clavier et souris
* 1 PC Windows, Mac ou Linux
* 1 adaptateur Micro-SD
* Connexion internet filaire

## Programme:
### Jour 1 :
* Prise en main de la Jetson Nano
* Cas d'école : la classification d'image avec Tensorflow
* Cas d'école 2 : la détection d'objets avec Tensorflow et OpenCV
* Discussion autour de l'apprentissage : moyens et méthodes

### Jour 2 :
* Comparaison CPU - GPU sur la Jetson Nano
* Estimation de la consommation énergétique du module embarqué
* Open project et/ou configuration de la Jetson Nano 

# Flash de la jetson Nano
* Télécharger l'image de la carte SD officielle de NVIDIA : https://developer.nvidia.com/embedded/jetpack#install
* Flasher la carte SD préparée avec Etcher

## Préparation de la carte SD sur Windows : 
* Télécharger et installer SD Memory Card Formatter : https://www.sdcard.org/downloads/formatter_4/eula_windows/
* Sélectionner la carte SD
* Select "Quick format"
* Laisser "Volume label" vide
* Cliquer sur "Format"

## Préparation de la carte SD sur Mac 
* `diskutil list external | fgrep '/dev/disk'`
* `sudo diskutil partitionDisk /dev/disk<n> 1 GPT "Free Space" "%noformat%" 100%` avec <n> le numéro du disk correspondant à la carte SD
* Télécharget et installer Etcher : https://www.balena.io/etcher
* Sélectionner l'image zippée
* Insérer la carte SD et ignorer l'éventuel message d'erreur

## Préparation de la carte SD sur Linux
* Télécharget et installer Etcher : https://www.balena.io/etcher
* Sélectionner l'image zippée
* Insérer la carte SD et ignorer l'éventuel message d'erreur

# Premier démarrage de la Jetson Nano
* Insérer la carte SD dans le slot de la Jetson
* Connecter l'écran, un clavier et une souris
* Alimenter la Jetson avec un cable Micro-USB ou une alimentation externe (jumper à modifier sur la carte pour sélectionner l'alimentation)


# Configuration de la jetson Nano
* Utiliser les capacités d'alimentation maximale : `sudo nvpmodel -m 0` puis `sudo jetson_clocks`
* Faisons de la place : `sudo apt-get purge libreoffice*`, `sudo apt-get clean`
* Update du système : `sudo apt update && sudo apt upgrade`
* `sudo reboot`

# Installation des dépendances nécessaires
```bash
sudo apt install nano screen
sudo apt-get install build-essential cmake git unzip pkg-config libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev libcanberra-gtk* python3-dev python3-numpy python3-pip libxvidcore-dev libx264-dev libgtk-3-dev libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev v4l-utils libavresample-dev libvorbis-dev libxine2-dev libfaac-dev libmp3lame-dev libtheora-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenblas-dev libatlas-base-dev libblas-dev liblapack-dev libeigen3-dev gfortran libhdf5-dev protobuf-compiler libprotobuf-dev libgoogle-glog-dev libgflags-dev
sudo apt-get install python3-pip 
sudo -H pip3 install -U jetson-stats
sudo apt-get install python-matplotlib
```

## Install virtualenv
```bash
sudo apt-get install virtualenv
python3 -m virtualenv -p python3 <chosen_venv_name>
source <chosen_venv_name>/bin/activate
```

## Install Tensorflow inside (virtualenv)
https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

```bash
pip3 install -U numpy grpcio absl-py py-cpuinfo psutil portpicker six mock requests gast astor termcolor protobuf keras-applications keras-preprocessing wrapt google-pasta setuptools testresources
env H5PY_SETUP_REQUIRES=0 pip install --no-binary=h5py h5py --no-build-isolation
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v46 tensorflow
```

```bash
nano ~/.bashrc
export OPENBLAS_CORETYPE=ARMV8
```

## Tester Tensorflow
```bash
source <chosen_venv_name>/bin/activate
python
import tensorflow as tf
print(tf.__version__)
```

## Installation de Jupyter notebook
* `sudo apt install jupyter`
* Générer un fichier de configuration : `jupyter notebook --generate-config`
* Ajouter à la fin du fichier `/home/cooptek/.jupyter/jupyter_notebook_config.py`:
```bash
import os
c = get_config()
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
c.Spawner.env.update('LD_PRELOAD')
```
* Pour lancer un nouveau notebook : `jupyter notebook`

## Installation d'OpenCV avec CUDA
```bash
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.4.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.4.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.5.4 opencv
mv opencv_contrib-4.5.4 opencv_contrib
rm opencv.zip
rm opencv_contrib.zip
```

```bash
cd ~/opencv
mkdir build
cd build
```

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -D WITH_OPENCL=OFF \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=5.3 \
    -D CUDA_ARCH_PTX="" \
    -D WITH_CUDNN=ON \
    -D WITH_CUBLAS=ON \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_NEON=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENMP=ON \
    -D WITH_OPENGL=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_TBB=ON \
    -D BUILD_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D BUILD_opencv_python3=TRUE \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_EXAMPLES=OFF ..
```

Increase swap memory
`sudo apt-get install dphys-swapfile`
Changer la limite du swap dans le fichier `/sbin/dphys-swapfile`: `CONF_MAXSWAP=4096`
Changer la valeur du swap dans le fichier `/etc/dphys-swapfile`: `CONF_SWAPSIZE=4096`

```bash
make -j1
sudo make install
```

```bash
sudo rm -rf ~/opencv
sudo rm -rf ~/opencv_contrib
```

## Vérifier les datasets Keras déjà téléchargée :
```bash
cd ~/.keras/datasets
```


```bash
pip install --user ipykernel

``