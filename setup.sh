#!/bin/sh

# Create datasets dir
mkdir -p data/{mnist, cifar10, svhn}

# Create virtual environment
python -m virtualenv venv

# Activate it
source venv/bin/Activate

# Install requirements
pip install -r requirements.txt