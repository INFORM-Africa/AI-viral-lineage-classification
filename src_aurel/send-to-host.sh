# ssh to the VM
# gcloud compute ssh aims-project-vm --project aims-ai2324-std-aurel-5ul
# ssh-add ~/.ssh/google_compute_engine	

VM="aims-project-vm-2"
DIR="~/aims-project"
SRCDIR="~/aims-project/source"


# Zip the models folder on the VM
gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "zip -r ~/aims-project/outputs/models_ouput.zip ~/aims-project/outputs/models/"

# Copy from the VM to the local machine
gcloud compute scp $VM:~/aims-project/outputs/models_ouput.zip /Users/masterbee/Desktop/AIMS-PROJECT

# Delete thezip file on the VM
gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "rm ~/aims-project/outputs/models_ouput.zip"