#!/usr/bin/env bash

set -o nounset                              # Treat unset variables as an error
set -e


vocabfile=data/tinyshakespeare/voc.txt
inmodel=models/char_lm_model.20180517

python ./src/sample_rnn_lm.py \
  --inmodel ${inmodel} \
  --vocabfile ${vocabfile} \
  --spell-word \
  --sample-sent
