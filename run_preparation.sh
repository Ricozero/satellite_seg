#!/bin/bash

# 脚本中多行字符串行尾若无"\"，回车会被当成空格
HELP_TEXT="""
参数：\n\
-c, --use_crf 可选，使用CRF\n\
-d, --dataset [crf|dstl] 必选，选择使用的数据集\n\
-h, --help 显示此帮助\
"""

DATASET=
CRF=0

if [ $# -lt 2 ];then
    echo "参数数量不足!"
    echo -e $HELP_TEXT
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
        -h | --help)
            echo -e $HELP_TEXT
            exit ;;
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