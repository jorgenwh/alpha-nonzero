#!/bin/bash

START_BATCH=66002
END_BATCH=100000

for ((i=START_BATCH; i<=END_BATCH; i++))
do
    echo "Annotating batch $i/$END_BATCH"
    python annotate.py $i
done
