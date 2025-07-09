# AI viral lineage classification project

This repository contains the code sources for the AI variant classification project and sub-projects.

van Zyl, D.J., Dunaiski, M., Tegally, H. et al. Alignment-free viral sequence classification at scale. BMC Genomics 26, 389 (2025). https://doi.org/10.1186/s12864-025-11554-5

van Zyl, D.J., Dunaiski, M., Tegally, H. et al. Craft: A Machine Learning Approach to Dengue Subtyping. bioRxiv, (2025). https://doi.org/10.1101/2025.02.10.637410

The repository also acts as a command-line tool for training and running custom viral subtyping models.

## Installation & Setup

1. Navigate to the src directory:
```bash
cd src
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the custom ai-viral package developed for viral sequence feature extraction and modelling:
```bash
pip install -e .
```

Once this is complete, you should be ready to begin modelling.

## Settings

The training and prediction scripts provided work on the basis of a ```settings.toml``` file defining the parameters to be used for preprocessing, feature extraction and modelling.

An example toml file ```dengue_standard.toml``` has been provided in the settings folder. Refer to the comments within the example toml for documentation regarding the different settings options.

## Training

To train a custom model, use the ```run_training.py``` script. Specify the settings to be used in an appropriate toml file and pass the path to the toml file as the only command-line parameter.

For example:
```bash
python run_training.py ../settings dengue_standard.toml  
```
Make sure to execute this command from the src folder. The path to the training data must be specified in the toml settings. Training data is expected in csv format with required columns:

Accession ID, Sequence, Target

Where Target is the lineage to be predicted by the model. If you have a large amount of data (>100,000) sequences, consider using the Dask setting to allow for efficient parallel processing.

## Prediction

To classify lineages using an existing model, use the ```run_prediction.py``` script. Again, specify the settings to be used in an appropriate toml file and pass the path to the toml file as the only command-line parameter. In the toml file, you will specify the name of the model to be used.

For example:
```bash
python run_training.py ../settings dengue_standard.toml  
```

Make sure to execute this command from the src folder. The path to the sequences must be specified in the toml settings. The sequence data is expected in fasta format.

## Running an Example

The standard dengue Craft model is provided already in the ```models``` folder and can be used to classify complete dengue sequences. 

Likewise, an example fasta file of publicly available dengue sequences has been provided in ```example_input/dengue_sample.fasta```.

The template ```dengue_standard.toml``` provided has been setup to run the example fasta through the provided Craft model. 

Running:
```bash
python run_training.py ../settings dengue_standard.toml  
```

Will provide the Craft predictions for the sample sequences in ```predictions/dengue_sample_BCGR_128_craft_predictions.csv```.

## Running Individual Scripts

The provided ```run_training.py``` and ```run_prediction.py``` scripts automate the preprocessing, feature extraction and modelling pipelines. To run processes individually, refer to the python scripts in the ```scripts``` directory.

## Contact

If you require any assistance or further information, you can contact me at danielvanzyl@sun.ac.za.