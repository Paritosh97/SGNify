#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }


echo -e "\nDownloading SGNify..."
wget --post-data "username=paritosh.sharmas@gmail.com&password=12345678" 'https://download.is.tue.mpg.de/download.php?domain=sgnify&resume=1&sfile=data.zip' -O 'data.zip' --no-check-certificate --continue
unzip data.zip
rm data.zip
mkdir data/coco17
mv data/vitpose-h-multi-coco.pth data/coco17/vitpose-huge.pth

GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive
conda env create -f environment.yml 
eval "$(conda shell.bash hook)"
git submodule update --init --recursive

cd spectre
echo -e "\nDownload pretrained SPECTRE model..."
gdown --id 1vmWX6QmXGPnXTXWFgj67oHzOoOmxBh6B
mkdir -p pretrained/
mv spectre_model.tar pretrained/

# add template mtl -> https://github.com/yfeng95/DECA/issues/67
FILE="data/template.mtl"
# Check if the file exists
if [ ! -f "$FILE" ]; then
    # File does not exist, create it and write the content
    echo -e "newmtl FaceTexture\nmap_Kd mean_texture.jpg" > $FILE
    echo "File $FILE created with the specified content."
else
    # File exists, overwrite it with the new content
    echo -e "newmtl FaceTexture\nmap_Kd mean_texture.jpg" > $FILE
    echo "File $FILE already existed and has been overwritten with the specified content."
fi
cd ..

cp -r data/FLAME2020 spectre/data/FLAME2020

conda activate sgnify
export PYTHONPATH=$(pwd)


# Create the target directory if it doesn't exist
mkdir -p ./spectre/external/face_detection/ibug/face_detection/retina_face/weights/
# URLs of the files to be downloaded
MOBILENET_URL="https://github.com/elliottzheng/face-detection/releases/download/0.0.1/mobilenet0.25_Final.pth"
RESNET_URL="https://github.com/elliottzheng/face-detection/releases/download/0.0.1/Resnet50_Final.pth"
# Download the files using wget
wget $MOBILENET_URL -O mobilenet0.25_Final.pth
wget $RESNET_URL -O Resnet50_Final.pth
