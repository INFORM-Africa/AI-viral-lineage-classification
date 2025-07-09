import argparse

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process and analyze genomic sequences based on various feature extraction methods.')

    # Feature type
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Viral Dataset name.'
    )
    
    parser.add_argument(
        '--feature',
        type=str.upper,
        choices=['FCGR', 'ACS', 'KMER', 'SWF', 'GSP', 'MASH', 'RTD'],
        required=True,
        help='Type of feature extraction to perform. Choices include FCGR, ACS, kmer, SWF, GSP, Mash, and RTD.'
    )

    # Run name for output folder
    parser.add_argument(
        '--run_name',
        type=str,
        required=True,
        help='Name for this run, used to create the output folder name.'
    )

    # Degenerate nucleotides handling
    parser.add_argument(
        '--deg',
        choices=['remove', 'replace'],
        required=True,
        help='Specifies how degenerate nucleotides should be handled: removed or replaced randomly.'
    )

    # Word length (k)
    parser.add_argument(
        '--k',
        type=int,
        help='Specifies the word length for kmer, Mash, SWF, and RTD feature extraction.'
    )

    # Image resolution for FCGR
    parser.add_argument(
        '--res',
        type=int,
        choices=[32, 64, 128, 256],
        help='Specifies the image resolution for FCGR feature extraction.'
    )

    # Chaos game mode for FCGR
    parser.add_argument(
        '--mode',
        choices=['FCGR', 'BCGR'],
        default='FCGR',
        help='Specifies the chaos game mode: FCGR (Frequency) or BCGR (Binary). Default is FCGR.'
    )

    # Sketch size for Mash
    parser.add_argument(
        '--size',
        type=int,
        choices=[1000, 2000],
        help='Specifies the sketch size for Mash feature extraction.'
    )

    # Numeric mapping for GSP
    parser.add_argument(
        '--map',
        choices=['real', 'eiip', 'justa', 'pp'],
        help='Specifies the form of numeric mapping for GSP feature extraction.'
    )

    # Spaced pattern for SWF
    parser.add_argument(
        '--pattern',
        type=str,
        help='Specifies the spaced pattern for SWF feature extraction.'
    )

    return parser

def validate_args(args, parser):
    """Validate the inter-dependency of command line arguments based on the feature type."""
    feature_needs = {
        'FCGR': ('res', 'mode'),
        'KMER': ('k',),
        'SWF': ('k'),
        'GSP': ('map',),
        'MASH': ('k', 'size'),
        'RTD': ('k',)
    }

    needed = feature_needs.get(args.feature, ())
    missing = [arg for arg in needed if getattr(args, arg) is None]

    if missing:
        parser.error(f"Feature {args.feature.upper()} requires the following missing arguments: {', '.join(missing)}") 