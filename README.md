# IPE / C-IPE Experiments
Experiments to compare IPE (Individual Prescription Effect)
and C-IPE (Complianca aware Individual Prescription Effect) approaches on synthetical and real datasets.

* IPE: `P(Y=1|P=1,X=x) - P(Y=1|P=0,X=x)`
* C-IPE: `P(M=1|P=1,X=x) * (P(Y=1|M=1,X=x) - P(Y=1|M=0,X=x))`
* where
  * `Y` is the outcome
  * `P` is the treatment prescription
  * `T` is the evidence of treatment acceptation
  * `X` are individual features


## Reproducing Experiments

### Requirements
  * python >= 3.6
  * numpy
  * pandas
  * scikit-learn
  * tqdm

To install:

```
$ pip install -r requirements.txt
```

### On synthetic datasets
To run experiments on one dataset with multiple compliance values:
```
$ python run_synthetic_dataset.py multi ./my_synth_results
```
We may want to run them on multiple CPUs:
```
$ python run_synthetic_dataset.py multi ./my_synth_results --ncpus 30
```

To run experiments on several datasets with different constant compliance values:
```
$ python run_synthetic_dataset.py single ./my_synth_results
$ python run_synthetic_dataset.py single ./my_synth_results --ncpus 30
```

You should obtain the same results as in `synth_results` folder.

### On CRITEO-UPLIFT1 Dataset
#### Get and convert the dataset
Download [`criteo-uplift-v2.csv.gz`](http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz)
(311,422,618 bytes), i.e. the un-biased version presented in "erratum" section of
https://ailab.criteo.com/criteo-uplift-prediction-dataset/.

(Optional) You may want to convert it into a pickle file to speed-up the experiments.
```
$ python run_criteo_dataset.py convert ./criteo-uplift-v2.csv.gz ./criteo-uplift-v2.pkl.gz
```
Or directly:
```
$ python run_criteo_dataset.py convert http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz ./criteo-uplift-v2.pkl.gz
```

### Run experiments

To run the experiments, with one configuration of hyperparameters (L2 penalty with C=10):
```
$ python run_criteo_dataset.py run ./criteo-uplift-v2.pkl.gz ./my_criteo_results
$ python run_criteo_dataset.py run ./criteo-uplift-v2.pkl.gz ./my_criteo_results --ncpus 30
```

To explore the complete grid of hyperparameters:
```
$ python run_criteo_dataset.py run ./criteo-uplift-v2.pkl.gz ./my_criteo_results --fullgrid --ncpus 30
```

You should obtain the same results as in `criteo_results` folder.

## Analysing Results

You need [jupyter](https://jupyter.readthedocs.io/en/latest/install.html)
and [matplotlib](https://matplotlib.org/users/installing.html).

To display the results, run cells in notebook `analyse_results.ipynb`.
