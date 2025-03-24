#!/bin/bash

# # Execute feature extraction commands
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg remove --k 2
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg remove --k 3
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg remove --k 4
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg remove --k 5
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg remove --k 6
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg replace --k 7
# python extract_features.py --dataset SARS-CoV-2 --feature KMER --deg replace --k 7
# python extract_features.py --dataset SARS-CoV-2 --feature MASH --deg replace --k 21 --size 1000

# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg replace --k 5
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg replace --k 6
#python extract_features.py --dataset SARS-CoV-2 --feature MASH --deg remove --res 128

# python extract_features.py --dataset SARS-CoV-2 --feature kmer --deg remove --k 2
# python extract_features.py --dataset SARS-CoV-2 --feature kmer --deg remove --k 3
# python extract_features.py --dataset SARS-CoV-2 --feature kmer --deg remove --k 4
# python extract_features.py --dataset SARS-CoV-2 --feature SWF --deg remove --k 7

# python extract_features.py --dataset SARS-CoV-2 --feature RTD --deg remove --k 2
# python extract_features.py --dataset SARS-CoV-2 --feature RTD --deg remove --k 3
# python extract_features.py --dataset SARS-CoV-2 --feature RTD --deg remove --k 4
# python extract_features.py --dataset SARS-CoV-2 --feature RTD --deg replace --k 7
# python extract_features.py --dataset SARS-CoV-2 --feature RTD --deg replace --k 4

# python extract_features.py --dataset SARS-CoV-2 --feature GSP --deg replace --map real
# python extract_features.py --dataset SARS-CoV-2 --feature FCGR --deg replace --res 128

# python extract_features.py --dataset SARS-CoV-2 --feature kmer --deg replace --k 5
# python extract_features.py --dataset SARS-CoV-2 --feature kmer --deg replace --k 6
# python extract_features.py --dataset SARS-CoV-2 --feature kmer --deg remove --k 7
# python extract_features.py --dataset SARS-CoV-2 --feature fcgr --deg remove --res 32
# python extract_features.py --dataset SARS-CoV-2 --feature fcgr --deg remove --res 64
# python extract_features.py --dataset SARS-CoV-2 --feature fcgr --deg remove --res 128

# python extract_features.py --dataset SARS-CoV-2 --feature fcgr --deg replace --res 32
# python extract_features.py --dataset SARS-CoV-2 --feature fcgr --deg replace --res 64
python extract_features.py --dataset SARS-CoV-2 --feature fcgr --deg remove --res 128
# python extract_features.py --dataset SARS-CoV-2 --feature rtd --deg remove --k 5
# python extract_features.py --dataset SARS-CoV-2 --feature rtd --deg remove --k 6
# python extract_features.py --dataset SARS-CoV-2 --feature rtd --deg remove --k 7
# python extract_features.py --dataset SARS-CoV-2 --feature rtd --deg replace --k 5
# python extract_features.py --dataset SARS-CoV-2 --feature rtd --deg remove --k 5
# python extract_features.py --dataset SARS-CoV-2 --feature swf --deg remove --k 4
# python extract_features.py --dataset SARS-CoV-2 --feature rtd --deg replace --k 7
# python extract_features.py --dataset SARS-CoV-2 --feature gsp --deg remove --k 21 --size 1000
# python extract_features.py --dataset SARS-CoV-2 --feature mash --deg replace --k 30 --size 1000
# python extract_features.py --dataset SARS-CoV-2 --feature mash --deg replace --k 21 --size 1000