#!/bin/bash

DATADIR=data

mkdir -p $DATADIR

wget -O "$DATADIR/dow_top10_2005_2019.npy" "https://github.com/tylerkastner/distribution-equivalence/raw/main/option_trading/dow_top10_2005_2019.npy"