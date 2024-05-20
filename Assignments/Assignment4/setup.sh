#!/usr/bin/bash

# create virtual env
python -m venv env
# activate env
source ./env/bin/activate
# install requirements
sudo apt-get update
sudo apt-get install -y python3-opencv

pip install --upgrade pip
pip install -r requirements.txt

deactivate