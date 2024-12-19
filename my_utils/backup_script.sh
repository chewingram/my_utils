#!/bin/bash

# Paths and Variables
PROXYJUMP_SSH="-J gwceci"
PROXYJUMP_RSYNC="-e \"ssh -J gwceci\""
REMOTE_USER="slongo"
REMOTE_HOST="nic5.uliege.be"
REMOTE_DIR="/home/users/s/l/slongo/test_backup"
LOCAL_BACKUP_DIR="/Users/longosamuel/Work/backup_nic5"
WORKING_DIR="$LOCAL_BACKUP_DIR/backup_$(date +%Y_%m_%d_%H_%M)"
TIMESTAMP_FILE="$REMOTE_DIR/.last_backup_time"
REMOTE_MODIFIED_LIST="$REMOTE_DIR/remote_modified_files.txt"
LOCAL_MODIFIED_LIST="$WORKING_DIR/local_modified_files.txt"
LOCAL_MODIFIED_LIST_GZ="$WORKING_DIR/local_modified_files_gz.txt"

PREVIOUS_BACKUP=$(ls -1 -d $LOCAL_BACKUP_DIR/_backup_* | sort | tail -n 1)

# Check if a previous backup exists
if [ -z "$PREVIOUS_BACKUP" ]; then
    echo "No previous backup found."
    mkdir -p $WORKING_DIR
else
    echo "Previous backup found: $PREVIOUS_BACKUP"
    cp -r $PREVIOUS_BACKUP $WORKING_DIR
fi

# Step 1: Ensure the timestamp file exists on the remote machine
ssh $PROXYJUMP_SSH $REMOTE_USER@$REMOTE_HOST "[ ! -f $TIMESTAMP_FILE ] && touch $TIMESTAMP_FILE"

# Step 2: Create a list of modified or new files on the remote machine
ssh $PROXYJUMP_SSH $REMOTE_USER@$REMOTE_HOST "find $REMOTE_DIR -type f -newer $TIMESTAMP_FILE -printf '%P\n' > $REMOTE_MODIFIED_LIST"

# Step 3: Download the list of modified files
eval rsync $PROXYJUMP_RSYNC -avzP $REMOTE_USER@$REMOTE_HOST:$REMOTE_MODIFIED_LIST $LOCAL_MODIFIED_LIST
# Step 4: Map to local compressed files
# Convert the remote file paths into local .gz file paths
while read -r file; do
    echo "$file.gz" >> $LOCAL_MODIFIED_LIST_GZ
done < $LOCAL_MODIFIED_LIST

# Step 5: Unzip only the files in the local modified list
# Iterate through the local modified list and unzip files
while read -r file; do
    if [ -f "$file" ]; then
        gunzip "$file"
    fi
done < $LOCAL_MODIFIED_LIST_GZ

# Step 6: Rsync to update uncompressed files in the working directory
eval rsync $PROXYJUMP_RSYNC -avzP --files-from=$LOCAL_MODIFIED_LIST $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR $WORKING_DIR
ssh $PROXYJUMP_SSH $REMOTE_USER@$REMOTE_HOST "rm $REMOTE_MODIFIED_LIST"

# Step 7: Recompress only the updated files
while read -r file; do
    if [ -f "$WORKING_DIR/${file%}" ]; then
        gzip -f "$WORKING_DIR/${file%.gz}"
    fi
done < $LOCAL_MODIFIED_LIST

# Step 8: Update the reference timestamp
ssh $PROXYJUMP_SSH $REMOTE_USER@$REMOTE_HOST "touch $TIMESTAMP_FILE"

rm "$WORKING_DIR/remote_modified_files.txt.gz"
rm "$WORKING_DIR/local_modified_files_gz.txt"
# Cleanup (optional)
#rm $REMOTE_MODIFIED_LIST $LOCAL_MODIFIED_LIST
