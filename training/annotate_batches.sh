#!/bin/bash

START_BATCH=6
END_BATCH=10

for ((i=START_BATCH; i<=END_BATCH; i++))
do
    echo "Annotating batch $i/$END_BATCH"
    python annotate.py $i
done
