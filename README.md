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

Python3.5 or above is required because of type hinting.

```sh
$ mkdir venvdir
$ python3 -m venv venvdir
$ . ./venvdir/bin/activate
(venvdir) $ pip install -r requirements.txt
```

# Execution

Pass the date to fit to as a command line argument.  
For example when you pass `20130606`,
censored linear model will fit to `bidimpclk.20130606.sim2.csv`  
and the model will be evaluated using `bidimpclk.20130607.sim2.csv`.

```sh
(venvdir) $ cd ..
(venvdir) $ python -m winning_price_pred.censored_reg 20130606
Reading /mnt/data/nishioka/winning_price_pred/data/bidimpclk.20130606.sim2.csv for training ...
Reading /mnt/data/nishioka/winning_price_pred/data/bidimpclk.20130607.sim2.csv for test ...
Generating X_all for training ...
Generating X_win for training ...
Generating X_all for test ...
Generating X_win for test ...
Generating X_lose for test ...
Fitting LinearModel (l2reg=10) ...
MSE on all: 3410.568823079685, r2score on all: 0.036118986646214934
MSE on win: 808.4110845121388, r2score on win: 0.1458832121224426
MSE on lose: 17488.77158266221, r2score on lose: -7.808129940047861
Fitting CensoredLinearModel (l2reg=1) ...
MSE on all: 2080.861078965126, r2score on all: 0.41191555148556097
MSE on win: 1007.2580078911672, r2score on win: -0.06420605895469134
MSE on lose: 7889.272111806644, r2score on lose: -2.9733913594067714
Predicting by MixtureModel...
MSE on all: 2505.285576316084, r2score on all: 0.2919664357162428
MSE on win: 857.4402981611596, r2score on win: 0.09408199950134655
MSE on lose: 11420.463775719254, r2score on lose: -4.751857895603674

    BiddingPrice  NewBiddingPrice  PayingPrice is_win  PredPriceLM  PredPriceCLM  PredPriceMix
1            300            150.0          121   True    58.811314     80.684565     61.925462
2            300            150.0           94   True    44.387071     57.347588     44.838584
3            238            119.0           36   True    24.715549     35.880261     25.248636
4            300            150.0           44   True    21.888623     32.380057     21.993302
5            238            119.0          126  False    74.805937    110.473094     88.824603
6            227            113.5          154  False    66.329831     71.406766     66.726544
7            300            150.0           46   True    43.854636     56.200373     44.127768
8            238            119.0           65   True    27.216636     42.544953     28.527280
9            300            150.0           43   True    36.946535     47.412197     37.063628
10           300            150.0            9   True     7.403345     35.392005      8.190282
11           238            119.0          111   True    47.391180     48.556372     47.430856
12           300            150.0           44   True    27.149674     33.069704     27.202039
13           238            119.0           92   True    48.561989     65.341883     50.915033
14           238            119.0           57   True    50.646631     94.628123     61.729563
15           238            119.0           34   True    59.555921     95.535899     69.922451
16           238            119.0           46   True    19.044949     37.400270     19.652746
17           238            119.0           90   True    41.029389     50.631818     41.717587
18           300            150.0           17   True    29.265943     37.213748     29.297658
19           227            113.5           31   True    30.222903     29.959403     30.217675
20           300            150.0          106   True    70.320781     72.004427     70.375140
```

# Disclaimer

This library does not completely simulate the same result of KDD2015wpp,  
because the implementation is slightly different from that of R.
