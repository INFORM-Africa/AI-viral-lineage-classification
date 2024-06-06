#!/bin/bash

# ssh to the VM
# gcloud compute ssh aims-project-vm --project aims-ai2324-std-aurel-5ul
# ssh-add ~/.ssh/google_compute_engine	

# Function to display usage message
usage() {
  echo "Usage: $0 -n VM_ID"
  exit 1
}

# Parse command line options
while getopts ":n:" opt; do
  case $opt in
    n)
      VM_ID=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Check if VM_ID is set
if [ -z "$VM_ID" ]; then
  echo "VM_ID is required"
  usage
fi

# Define variables
VM="aims-project-vm-$VM_ID"
DIR="~/aims-project"
SRCDIR="~/aims-project/source"


# Zip the models folder on the VM
gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "zip -r ~/aims-project/outputs/models_ouput.zip ~/aims-project/outputs/models/"

# Copy from the VM to the local machine
gcloud compute scp $VM:~/aims-project/outputs/models_ouput.zip /Users/masterbee/Desktop/AIMS-PROJECT

# Delete thezip file on the VM
gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "rm ~/aims-project/outputs/models_ouput.zip"