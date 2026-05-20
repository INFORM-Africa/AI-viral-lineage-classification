#!/usr/bin/env python3
"""
AI Viral Lineage Classification - Prediction Runner

This script handles the complete prediction pipeline including:
1. Loading configuration from TOML files
2. Preprocessing input data (CSV/FASTA to parquet)
3. Feature extraction
4. Model prediction

Usage:
    python run_prediction.py path/to/config.toml
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import toml
from typing import Dict, Any


def load_toml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a TOML file.
    
    Args:
        config_path (str): Path to the TOML configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except toml.TomlDecodeError as e:
        print(f"Error: Invalid TOML format in '{config_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        sys.exit(1)


def determine_file_type(file_path: str) -> str:
    """
    Determine if the input file is CSV or FASTA based on extension.
    
    Args:
        file_path (str): Path to the input file
        
    Returns:
        str: 'csv' or 'fasta'
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.csv':
        return 'csv'
    elif extension in ['.fasta', '.fa', '.fas']:
        return 'fasta'
    else:
        print(f"Error: Unsupported file type '{extension}'. Expected .csv, .fasta, .fa, or .fas")
        sys.exit(1)


def run_preprocessing(config: Dict[str, Any]) -> None:
    """
    Run preprocessing based on the configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    data_path = config.get('DATA_PATH')
    dask_enabled = config.get('DASK', 0)
    
    if not data_path:
        print("Error: DATA_PATH not specified in configuration")
        sys.exit(1)
    
    # Check if the file exists
    if not os.path.exists(data_path):
        print(f"Error: Input file '{data_path}' not found")
        sys.exit(1)
    
    # Determine file type
    file_type = determine_file_type(data_path)
    
    # Get the script directory
    script_dir = Path(__file__).parent / 'scripts' / 'preprocessing'
    
    # Choose the appropriate preprocessing script
    if file_type == 'csv':
        script_path = script_dir / 'process_csv.py'
    else:  # fasta
        script_path = script_dir / 'process_fasta.py'
    
    # Check if the script exists
    if not script_path.exists():
        print(f"Error: Preprocessing script '{script_path}' not found")
        sys.exit(1)
    
    # Run the preprocessing script
    print(f"Running {file_type.upper()} preprocessing...")
    print(f"Input file: {data_path}")
    print(f"Dask enabled: {dask_enabled}")
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path), data_path, str(dask_enabled)
        ], capture_output=True, text=True, check=True)
        
        print("Preprocessing completed successfully!")
        if result.stdout:
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during preprocessing: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        sys.exit(1)


def get_dataset_basename(data_path: str) -> str:
    """
    Extract the basename of the dataset from the DATA_PATH.
    
    Args:
        data_path (str): Path to the input file
        
    Returns:
        str: Basename of the dataset (filename without extension)
    """
    # Get only the filename without path and extension
    basename = Path(data_path).stem
    return basename


def build_feature_extraction_args(config: Dict[str, Any], dataset_basename: str) -> list:
    """
    Build the command line arguments for feature extraction.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        dataset_basename (str): Basename of the dataset
        
    Returns:
        list: List of command line arguments
    """
    args = [
        '--dataset', dataset_basename,
        '--feature', config.get('FEATURE', 'FCGR'),
        '--run_name', config.get('RUN_NAME', 'default_run'),
        '--deg', config.get('DEG', 'remove')
    ]
    
    # Add feature-specific arguments
    feature = config.get('FEATURE', 'FCGR')
    
    if feature in ['KMER', 'SWF', 'RTD']:
        args.extend(['--k', str(config.get('K', 6))])
    
    if feature == 'FCGR':
        args.extend(['--res', str(config.get('RES', 128))])
        args.extend(['--mode', config.get('MODE', 'FCGR')])
    
    if feature == 'MASH':
        args.extend(['--k', str(config.get('K', 6))])
        args.extend(['--size', str(config.get('SIZE', 1000))])
    
    if feature == 'GSP':
        args.extend(['--map', config.get('MAP', 'real')])
    
    if feature == 'SWF':
        args.extend(['--pattern', config.get('PATTERN', '1101')])
    
    return args


def run_feature_extraction(config: Dict[str, Any]) -> None:
    """
    Run feature extraction based on the configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    data_path = config.get('DATA_PATH')
    if not data_path:
        print("Error: DATA_PATH not specified in configuration")
        sys.exit(1)
    
    # Get dataset basename
    dataset_basename = get_dataset_basename(data_path)
    
    # Get the feature extraction script path
    script_dir = Path(__file__).parent / 'scripts' / 'feature_extraction'
    script_path = script_dir / 'extract_features.py'
    
    # Check if the script exists
    if not script_path.exists():
        print(f"Error: Feature extraction script '{script_path}' not found")
        sys.exit(1)
    
    # Build command line arguments
    feature_args = build_feature_extraction_args(config, dataset_basename)
    
    # Run the feature extraction script
    print(f"Running feature extraction...")
    print(f"Dataset: {dataset_basename}")
    print(f"Feature type: {config.get('FEATURE', 'FCGR')}")
    print(f"Run name: {config.get('RUN_NAME', 'default_run')}")
    
    try:
        cmd = [sys.executable, str(script_path)] + feature_args
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Feature extraction completed successfully!")
        if result.stdout:
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during feature extraction: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        sys.exit(1)


def build_prediction_args(config: Dict[str, Any], dataset_basename: str) -> list:
    """
    Build the command line arguments for prediction.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        dataset_basename (str): Basename of the dataset
        
    Returns:
        list: List of command line arguments
    """
    args = [
        '--dataset', dataset_basename,
        '--run_name', config.get('RUN_NAME', 'default_run'),
        '--model_name', config.get('MODEL_NAME', 'random_forest_model')
    ]
    
    # Add Dask-specific arguments if using Dask
    if config.get('DASK', 0) == 1:
        args.extend(['--n_workers', str(config.get('N_WORKERS', 1))])
    
    return args


def build_segment_prediction_args(config: Dict[str, Any], dataset_basename: str) -> list:
    """
    Build the command line arguments for segment prediction.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        dataset_basename (str): Basename of the dataset
        
    Returns:
        list: List of command line arguments
    """
    args = [
        '--dataset', dataset_basename,
        '--run_name', config.get('RUN_NAME', 'default_run'),
        '--model_name', config.get('MODEL_NAME', 'random_forest_model'),
        '--res', str(config.get('RES', 128)),
        '--criterion', config.get('CRITERION', 'entropy'),
        '--class_weight', config.get('CLASS_WEIGHT', 'balanced')
    ]
    
    return args


def run_prediction(config: Dict[str, Any]) -> None:
    """
    Run prediction based on the configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    data_path = config.get('DATA_PATH')
    dask_enabled = config.get('DASK', 0)
    segment_enabled = config.get('SEGMENT', 0)
    
    if not data_path:
        print("Error: DATA_PATH not specified in configuration")
        sys.exit(1)
    
    # Get dataset basename
    dataset_basename = get_dataset_basename(data_path)
    
    # Choose the appropriate prediction script
    if segment_enabled == 1:
        script_name = 'predict_segment.py'
    elif dask_enabled == 1:
        script_name = 'predict_dask.py'
    else:
        script_name = 'predict_standard.py'
    
    # Get the prediction script path
    script_dir = Path(__file__).parent / 'scripts' / 'modelling'
    script_path = script_dir / script_name
    
    # Check if the script exists
    if not script_path.exists():
        print(f"Error: Prediction script '{script_path}' not found")
        sys.exit(1)
    
    # Build command line arguments
    if segment_enabled == 1:
        prediction_args = build_segment_prediction_args(config, dataset_basename)
    else:
        prediction_args = build_prediction_args(config, dataset_basename)
    
    # Run the prediction script
    print(f"Running {script_name}...")
    print(f"Dataset: {dataset_basename}")
    print(f"Model name: {config.get('MODEL_NAME', 'random_forest_model')}")
    print(f"Dask enabled: {dask_enabled}")
    print(f"Segment enabled: {segment_enabled}")
    
    try:
        cmd = [sys.executable, str(script_path)] + prediction_args
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print(f"{script_name} completed successfully!")
        if result.stdout:
            print(result.stdout)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during prediction: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        sys.exit(1)


def main():
    """
    Main function to handle the prediction pipeline.
    """
    parser = argparse.ArgumentParser(
        description='AI Viral Lineage Classification - Prediction Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_prediction.py settings/dengue_standard.toml
    python run_prediction.py my_custom_config.toml
        """
    )
    
    parser.add_argument(
        'config_path',
        help='Path to the TOML configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config_path}")
    config = load_toml_config(args.config_path)
    
    # Display key configuration
    print("\nConfiguration loaded:")
    print(f"  DATA_PATH: {config.get('DATA_PATH', 'Not specified')}")
    print(f"  FEATURE: {config.get('FEATURE', 'Not specified')}")
    print(f"  RUN_NAME: {config.get('RUN_NAME', 'Not specified')}")
    print(f"  DASK: {config.get('DASK', 0)}")
    print(f"  SEGMENT: {config.get('SEGMENT', 0)}")
    
    # Check if segment mode is enabled
    segment_enabled = config.get('SEGMENT', 0)
    
    if segment_enabled == 1:
        # Skip preprocessing and feature extraction for segment mode
        print("\n" + "="*50)
        print("Segment mode enabled - skipping preprocessing and feature extraction")
        
        # Run segment prediction directly
        print("\n" + "="*50)
        run_prediction(config)
    else:
        # Run normal pipeline: preprocessing -> feature extraction -> prediction
        print("\n" + "="*50)
        run_preprocessing(config)
        
        # Run feature extraction
        print("\n" + "="*50)
        run_feature_extraction(config)
        
        # Run prediction
        print("\n" + "="*50)
        run_prediction(config)
    
    print("\n" + "="*50)
    print("Prediction completed successfully!")
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
