#!/bin/bash
set -e
source ./.env
cd data
wget "https://$STORAGE_CREDS@u184584.your-storagebox.de/test.tar.gz"
wget "https://$STORAGE_CREDS@u184584.your-storagebox.de/IDRND_FASDB_val.tar.gz"
wget "https://$STORAGE_CREDS@u184584.your-storagebox.de/IDRND_FASDB_train.tar.gz"
tar -xvzf test.tar.gz
tar -xvzf IDRND_FASDB_val.tar.gz
tar -xvzf IDRND_FASDB_train.tar.gz
rm test.tar.gz
rm IDRND_FASDB_val.tar.gz
rm IDRND_FASDB_train.tar.gz
