winning price prediction in Python
===

Python implmentation of winning price prediction[1].
Original R implmentation is [here](https://github.com/wush978/KDD2015wpp).

[1] Wush Chi-Hsuan Wu, Mi-Yen Yeh, and Ming-Syan Chen. 2015. Predicting Winning Price in Real Time Bidding with Censored Data. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15). ACM, New York, NY, USA, 1305-1314. DOI: https://doi.org/10.1145/2783258.2783276

# Preparation

## Generate `bidimpclk.[yyyymmdd].sim2.Rds`

Follow [README.md of KDD2015wpp](https://github.com/wush978/KDD2015wpp/blob/master/README.md).

## Clone this repository

```sh
$ git clone https://github.com/ysk24ok/winning_price_pred.git
$ cd winning_price_pred/
```

## Convert .Rds to .csv

Convert `bidimpclk.[yyyymmdd].sim2.Rds` to `bidimpclk.[yyyymmdd].sim2.csv` to enable Python to read the dataset.

```sh
$ Rscript GenCsv.R [path to KDD2015wpp directory]
```

## Create venv and install dependencies

Python3.5 or above is required because this uses type hinting.

```sh
$ mkdir venvdir
$ python3 -m venv venvdir
$ . ./venvdir/bin/activate
(venvdir) $ pip install -r requirements.txt
```

## Run

Pass the date to fit to as a command line argument.  
For example when you pass `20130606`,
censored linear model will fit to `bidimpclk.20130606.sim2.csv`  
and the model will be evaluated using `bidimpclk.20130607.sim2.csv`.

```sh
(venvdir) $ cd ..
(venvdir) $ python -m winning_price_pred.censored_reg 20130606
```

The output will be as follows.

```sh
Reading /tmp/winning_price_pred/data/bidimpclk.20130606.sim2.csv for training ...
Reading /tmp/winning_price_pred/data/bidimpclk.20130606.sim2.csv for test ...
Generating win bids for training ...
Generating lose bids for training ...
Fitting CensoredLinearModel ...
Predicting ...
MSE: 2428.7938001167972
```

# Disclaimer

This library does not completely simulate the same result of KDD2015wpp,  
because the implementation is slightly different from that of R.
