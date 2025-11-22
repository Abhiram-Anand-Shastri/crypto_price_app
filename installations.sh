#!/bin/sh

# Upgrade to pip3
python3 -m pip install --upgrade pip  

# Install pipenv
sudo -H pip3 install -U pipenv 

# Create virtual environment
pipenv --python 3

# Activate virtual environment
pipenv shell 

# Install libraries
# pipenv install streamlit pandas matplotlib seaborn bs4 cryptocmd 