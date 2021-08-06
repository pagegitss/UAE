#!/bin/bash
set -ex
pushd datasets
wget https://www.dropbox.com/s/akviv6e9xi0tl00/Vehicle__Snowmobile__and_Boat_Registrations.csv
wget https://www.dropbox.com/s/ft7r9hqsi1xsc0h/census.csv
wget https://www.dropbox.com/s/itltxvxsndfii7l/cup98.csv
popd