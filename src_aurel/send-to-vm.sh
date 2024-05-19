# ssh to the VM
# gcloud compute ssh aims-project-vm --project aims-ai2324-std-aurel-5ul --zone europe-central2-c

# Copy file requirements.txt to the VM
#gcloud compute scp src_aurel/requirements.txt aims-project-vm:

DIR="~/aims-project"
SRCDIR="~/aims-project/source"

# Copy the project folder to the VM
zip -r "source.zip" ./*

# ssh-add ~/.ssh/google_compute_engine

# Delete the source folder on the VM
gcloud compute ssh aims-project-vm --project aims-ai2324-std-aurel-5ul --zone europe-central2-c --command "if [ -d "$SRCDIR" ]; then rm -r "$SRCDIR"; fi"
gcloud compute ssh aims-project-vm --project aims-ai2324-std-aurel-5ul --zone europe-central2-c --command "mkdir $SRCDIR"

# Copy the project folder to the VM
gcloud compute scp source.zip aims-project-vm:$SRCDIR

# Remove the zip file
rm source.zip

# Unzip the project folder on the VM
gcloud compute ssh aims-project-vm --project aims-ai2324-std-aurel-5ul --zone europe-central2-c --command "unzip -o $SRCDIR/source.zip -d $SRCDIR"