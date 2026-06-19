#!/bin/bash

set -e

for SEQLEN in 32 64 128 256 512 896; do
    echo "profile SEQLEN=$SEQLEN"
    (time python main.py ./models.cache/gpt2-small @./long-prompt.txt --debug -m 128 --truncate-to $SEQLEN) 2>&1
    echo 
done
