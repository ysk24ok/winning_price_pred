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
Reading /tmp/winning_price_pred/data/bidimpclk.20130606.sim2.csv for training ...
Reading /tmp/winning_price_pred/data/bidimpclk.20130607.sim2.csv for test ...
Generating X_win for training ...
Generating X_lose for training ...
Generating X_all for test ...
Generating X_win for test ...
Generating X_lose for test ...
Fitting LinearModel (l2reg=100) ...
MSE on all: 3292.466036216859, r2score on all: 0.06949671329137164
MSE on win: 729.1126283627675, r2score on win: 0.229665020595324
MSE on lose: 17160.729472123054, r2score on lose: -7.64291321674747
Fitting CensoredLinearModel (l2reg=100) ...
MSE on all: 2597.088240297978, r2score on all: 0.26602154224608476
MSE on win: 880.1973241496042, r2score on win: 0.07003834360488337
MSE on lose: 11885.817388181165, r2score on lose: -4.986230851260474
Predicting by MixtureModel...
MSE on all: 2885.042146719021, r2score on all: 0.1846411867927471
MSE on win: 710.6664005538678, r2score on win: 0.249154155698046
MSE on lose: 14648.857297633978, r2score on lose: -6.377821703537689

    BiddingPrice  NewBiddingPrice  PayingPrice is_win  PredPriceLM  PredPriceCLM  PredPriceMix
1            300            150.0          121   True    55.908983     68.553728     57.709246
2            300            150.0           94   True    46.488827     58.084333     46.892786
3            238            119.0           36   True    22.919368     68.662003     25.103462
4            300            150.0           44   True    20.929469     52.556243     21.245026
5            238            119.0          126  False    65.138778     86.994898     73.729136
6            227            113.5          154  False    64.546913     69.410512     64.926956
7            300            150.0           46   True    46.517580     59.659691     46.808330
8            238            119.0           65   True    27.634903     55.535481     30.020536
9            300            150.0           43   True    38.192278     61.344259     38.451309
10           300            150.0            9   True    13.173031     59.828159     14.484799
11           238            119.0          111   True    42.302315     82.568356     43.673405
12           300            150.0           44   True    21.995282     46.638464     22.213258
13           238            119.0           92   True    47.762757     61.379385     49.672216
14           238            119.0           57   True    57.391411     72.014857     61.076385
15           238            119.0           34   True    56.841204     69.548098     60.502307
16           238            119.0           46   True    26.281604     51.343687     27.111481
17           238            119.0           90   True    39.592026     66.175959     41.497275
18           300            150.0           17   True    31.644904     51.268748     31.723212
19           227            113.5           31   True    32.738627     55.368265     33.187611
20           300            150.0          106   True    74.008133     79.877595     74.197638
```

# Disclaimer

This library does not completely simulate the same result of KDD2015wpp,  
because the implementation is slightly different from that of R.
