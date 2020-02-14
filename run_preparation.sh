#!/bin/bash

DATASET=
CRF=0

if [ $# -lt 2 ];then
    echo "Not enough arguments."
    exit
fi

while [ -n "$1" ]
do
    case $1 in
        -c | --use_crf)
            CRF=1 ;;
        -d | --dataset)
            if [ $2 = ccf ];then
                DATASET=ccf
                shift
            elif [ $2 = dstl ];then
                DATASET=dstl
                shift
            else
                echo $2: No such a dataset
                exit
            fi ;;
        *)
            echo $1: Unknown option
            exit ;;
    esac
    shift
done

echo "Using dataset $DATASET."
if [ $CRF -eq 1 ];then
    echo "Using CRF."
fi

if [ $DATASET = ccf ];then
    echo "Generating CCF dataset..."
    python src/preparation/ccf.py $CRF
elif [ $DATASET = dstl ];then
    echo "Generating Dstl dataset..."
    python src/preparation/dstl.py $CRF
fi

echo "Done."