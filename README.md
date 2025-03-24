# AI viral lineage classification project

This repository contains the code sources for the AI variant classification project and sub-projects.

## Installation & Setup

1. Navigate to the source directory:
```bash
cd src
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode to use the library of feature extraction techniques:
```bash
pip install -e .
```

## Usage

Ensure that your data is located in a directory /data in the root of this project.

### Feature Extraction

1. Navigate to the feature extraction scripts:
```bash
cd scripts/02_feature_extraction
```

2. Configure the `extract_features.sh` script according to your needs. Specify the feature extraction technique(s) and related parameters. Then run:
```bash
./extract_features.sh
```

### Model Training

1. Navigate to the modeling scripts:
```bash
cd scripts/03_modelling
```

2. Configure the `train.sh` script according to your needs. Specify the feature extraction technique(s) for which the Random Forest models should be trained. Then run:
```bash
./train.sh
```
