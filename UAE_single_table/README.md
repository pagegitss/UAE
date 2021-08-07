# Source code for the single-table version of UAE/UAE-Q

## [A Unified Deep Model of Learning from both Data and Queries for Cardinality Estimation ](https://arxiv.org/pdf/2107.12295)

### Datasets Download

* DMV: The DMV dataset is publically available at [catalog.data.gov](https://catalog.data.gov/dataset/vehicle-snowmobile-and-boat-registrations). We use the frozen snapshot from [Naru](https://github.com/naru-project/naru) project.
* Census: The Census (or adult) dataset is publically available at [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). We use all the tuples (totally 48,842) of `adult.data` and `adult.test`.
* KDDCup98: The KDDCup98 dataset is also publically available at [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets/KDD+Cup+1998+Data). We use the learning dataset (~95K tuples) in `cup98lrn.zip` and remove the first row (headline).

You can download all the datasets under `./datasets` by runing 

```bash
bash ./download_datasets.sh
```

### Model Training of UAE 
Please run `python train_uae.py --help` to see a list of hyperparameters.

**Important Hyperparameters for Training UAE**:

`--run-uaeq`: whether to run the query-driven UAE-Q.

`--bs`: The batch size of data. The batch size of queries will be adjusted according to `--bs`. The principle is to ensure that the model goes through all the data and queries per epoch. Thus, we do not need to specify the batch size of queries when running UAE.

`--q-bs`: The batch size of queries. `--q-bs` only works for running UAE-Q.

`--constant-lr`: Constant learning rate. We turn on the `--constant-lr` for training the single-table version of UAE/UAE-Q, because we found it is more efficient and effective for UAE/UAE-Q training.

Examples for UAE:
```bash
python train_uae.py --num-gpus=1 --dataset=dmv --epochs=50 --constant-lr=5e-4 --bs=4096  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking

python train_uae.py --num-gpus=1 --dataset=census --epochs=50 --constant-lr=5e-4 --bs=100  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking

python train_uae.py --num-gpus=1 --dataset=cup98 --epochs=50 --constant-lr=5e-4 --bs=100  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking
```
Examples for UAE-Q:
```bash
python train_uae.py --num-gpus=1 --dataset=census --epochs=50 --constant-lr=5e-4 --q-bs=100 --run-uaeq  --residual --layers=2 --fc-hiddens=128 --direct-io --column-masking
```

### Model Testing of trained UAE 

Please run `python eval_model.py --help` to see a list of hyperparameters.

**Important Hyperparameters for Testing UAE**:

`--random-workload`: whether to evaluate the random workload.

Examples:
```bash
python eval_model.py --dataset=census --glob='uae-census-bs-100-20epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-2.pt'  --psample=200 --residual --direct-io --column-masking 

python eval_model.py --dataset=dmv --glob='uae-dmv-bs-4096-30epochs-psample-200-seed-0-tau-1.0-q-weight-0.0001-layers-2.pt' --psample=1000 --residual --direct-io --column-masking --random-workload
```




