# ssh to the VM
# gcloud compute ssh aims-project-vm --project aims-ai2324-std-aurel-5ul
# ssh-add ~/.ssh/google_compute_engine	

# Copy file requirements.txt to the VM
# gcloud compute scp src_aurel/requirements.txt aims-project-vm:

VM="aims-project-vm-1"
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

############################################################################
############################################################################

# Zip the models folder on the VM
# gcloud compute ssh $VM --project aims-ai2324-std-aurel-5ul --command "zip -r ~/aims-project/outputs/models_ouput.zip ~/aims-project/outputs/models"

# Copy from the VM to the local machine
# gcloud compute scp $VM:~/aims-project/outputs/models_ouput.zip /Users/masterbee/Desktop/AIMS-PROJECT