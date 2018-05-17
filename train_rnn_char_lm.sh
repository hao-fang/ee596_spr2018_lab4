#!/usr/bin/env bash

set -o nounset                              # Treat unset variables as an error
set -e

trainfile=data/tinyshakespeare/train.txt
validfile=data/tinyshakespeare/valid.txt
vocabfile=data/tinyshakespeare/voc.txt
outmodel=models/char_lm_model.20180517
nhidden=100
initalpha=0.01
initrange=0.05
batchsize=20
bptt=10

python ./src/train_rnn_lm.py \
  --trainfile ${trainfile} \
  --validfile ${validfile} \
  --vocabfile ${vocabfile} \
  --init-alpha ${initalpha} \
  --init-range ${initrange} \
  --batchsize ${batchsize} \
  --nhidden ${nhidden} \
  --outmodel ${outmodel} \
  --bptt ${bptt} \
  --validate

