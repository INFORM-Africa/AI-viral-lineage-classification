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

# Copy the project folder to the VM
zip -r "source.zip" ./*

# Delete the source folder on the VM
gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "if [ -d "$SRCDIR" ]; then rm -r "$SRCDIR"; fi"
gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "mkdir $SRCDIR"

# Copy the project folder to the VM
gcloud compute scp source.zip $VM:$SRCDIR

# Remove the zip file
rm source.zip

# Unzip the project folder on the VM
gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "unzip -o $SRCDIR/source.zip -d $SRCDIR"


# gcloud compute scp /Users/masterbee/Desktop/Archive.zip aims-project-vm-1:~/aims-project
# unzip -o ./Archive.zip -d ~/aims-project/outputs/models/reports