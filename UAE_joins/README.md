# Source code for the join version of UAE/UAE-Q

## [A Unified Deep Model of Learning from both Data and Queries for Cardinality Estimation ](https://arxiv.org/pdf/2107.12295)

### Dataset Download

Please follow the instructions from [Neurocard](https://github.com/neurocard/neurocard) project to download IMDB dataset in `./datasets`.

### Instruction for model training of UAE with job-light-ranges basic settings on MSCN workload
```
python run_uae.py --run job-light-ranges-mscn-workload
```

### Instruction for model testing of the pretrained UAE (with job-light-ranges basic settings) on subqueries of JOB-light
```
python run_uae.py --run uae-job-light-ranges-reload
```
If you want to evaluate UAE on the original JOB-light queries, simply change the value of 'job_light_queries_csv' at line 627 in `experiments.py` to './queries/job-light.csv'.

### New Hyper-parameters in `experiments.py`
train_virtual_cols: True: allow the virtual columns to be involved in query learning. False: fix the virtual columns during query learning.

run_uaeq: True: train query-driven uae-q from queries. False: train uae from both data and queries.

To change the the configuration for the run, please modify the corresponding section in `experiments.py`.

### Notes for hyper-parameter tuning
The current UAE model was trained using the column settings of job-light-ranges, and thus the hyper-parameters were tuned accordingly. If you want to train UAE mode using other column settings (e.g., job-light), please carefully tune the hyper-parameters (especially q_weight and warmups for UAE).

### Acknowledgment

This code is based on Neurocard. Thanks to the contributors of Neurocard.




