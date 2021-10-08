#!/bin/bash
set -ex

mkdir -p datasets/job && pushd datasets/job
wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz && tar -xvzf imdb.tgz && popd

python3 scripts/prepend_imdb_headers.py
