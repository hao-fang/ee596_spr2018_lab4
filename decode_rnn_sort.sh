#!/usr/bin/env bash

set -o nounset                              # Treat unset variables as an error
set -e


vocabfile=data/sorting_numbers/vocab
inmodel=models/sort_number_model.20180517

python ./src/decode_rnn_sort.py \
  --inmodel ${inmodel} \
  --vocabfile ${vocabfile}
