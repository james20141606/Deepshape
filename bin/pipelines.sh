#! /bin/bash
# Executable directory
BinDir=$(dirname $BASH_SOURCE)
# Auxialillary script directory
ScriptDir=$(dirname $BinDir)/scripts
if [ -f $BinDir/config ];then . $BinDir/config;fi
if [ -f $BinDir/functions ];then . $BinDir/functions;fi
if [ -f $BinDir/bashrc ];then . $BinDir/functions;fi

# default parameters
cutoff1=${cutoff1:=5}
cutoff2=${cutoff2:=95}
window_size=${window_size:=51}
batch_size=${batch_size:=50}
percentile=${percentile:=5-95}

_func_list_before=$(declare -F | cut -d' ' -f3)

set -e

train_test_split(){
    local data_name=$1

    echo "train_test_split: $data_name"
    awk 'NF==2{print $0}' data/icSHAPE/${data_name}.all.txt > data/icSHAPE/${data_name}.all.id_list.txt
    local n_transcripts=$(wc -l < data/icSHAPE/${data_name}.all.id_list.txt)
    local n_train=$(awk -v n=$n_transcripts 'BEGIN{print int(n*0.8)}')
    shuf data/icSHAPE/${data_name}.all.id_list.txt | head -n $n_train > data/icSHAPE/${data_name}.train.id_list.txt
    cat data/icSHAPE/${data_name}.all.id_list.txt data/icSHAPE/${data_name}.train.id_list.txt \
        | sort | uniq -u > data/icSHAPE/${data_name}.test.id_list.txt
    awk 'FNR==NR{sel[$0]=1;next}FNR!=NR{if(NF==2){name=$0} if(name in sel){print $0}}' \
        data/icSHAPE/${data_name}.train.id_list.txt \
        data/icSHAPE/${data_name}.all.txt \
        > data/icSHAPE/${data_name}.train.txt
    awk 'FNR==NR{sel[$0]=1;next}FNR!=NR{if(NF==2){name=$0} if(name in sel){print $0}}' \
        data/icSHAPE/${data_name}.test.id_list.txt \
        data/icSHAPE/${data_name}.all.txt \
        > data/icSHAPE/${data_name}.test.txt
}

prepare_icSHAPE(){
    local data_name=$1

    echo "prepare: $data_name"
    make_dir data/icSHAPE
    ls data/icSHAPE/${data_name} | awk '{split($0, a, ".txt");print a[1]}' > data/icSHAPE/${data_name}.all.id_list.txt
    awk -v data_name=$data_name '{print "data/icSHAPE/" data_name "/" $0 ".txt"}' data/icSHAPE/${data_name}.train.id_list.txt | xargs cat > data/icSHAPE/${data_name}.train.txt
    awk -v data_name=$data_name '{print "data/icSHAPE/" data_name "/" $0 ".txt"}' data/icSHAPE/${data_name}.test.id_list.txt | xargs cat > data/icSHAPE/${data_name}.test.txt
}

# get percentile from the whole dataset
get_percentile_icSHAPE(){
    local data_name=$1
    : ${percentile:?}
    local pct1=$(echo $percentile | awk '{split($0, a, /-/); print a[1]; exit 0}')
    local pct2=$(echo $percentile | awk '{split($0, a, /-/); print a[2]; exit 0}')

    echo "get_percentile: percentile=$percentile, $data_name"
    awk '{if((NF == 3) && ($3 != "NULL")) print $3}' data/icSHAPE/${data_name}.all.txt \
        | bin/percentile.py $pct1 $pct2 \
        > data/icSHAPE/${data_name}.p${percentile}.percentile.txt
}

create_dataset_icSHAPE(){
    local data_name=$1
    : ${percentile:?} ${window_size:?}
    local test_stride=${test_stride:=1}
    if [ "$dense" -eq 1 ];then
        local dense_opt="--dense"
        local dense_str=".dense"
        test_stride=$window_size
    fi

    echo "create_dataset: percentile=$percentile, window_size=$window_size, $data_name"
    local cutoff1=$(awk 'NR==1{print $2}' data/icSHAPE/${data_name}.p${percentile}.percentile.txt)
    local cutoff2=$(awk 'NR==2{print $2}' data/icSHAPE/${data_name}.p${percentile}.percentile.txt)
    bin/create_dataset.py icSHAPE $dense_opt --window-size $window_size --stride 5 \
        --icshape-cutoff1 $cutoff1 --icshape-cutoff2 $cutoff2 -i data/icSHAPE/${data_name}.train.txt \
        -o data/icSHAPE/${data_name}.p${percentile}.w${window_size}${dense_str}.train.h5
    bin/create_dataset.py icSHAPE $dense_opt --window-size $window_size --stride $test_stride \
        --icshape-cutoff1 $cutoff1 --icshape-cutoff2 $cutoff2 -i data/icSHAPE/${data_name}.test.txt \
        -o data/icSHAPE/${data_name}.p${percentile}.w${window_size}${dense_str}.test.h5
}

train_icSHAPE(){
    local data_name=$1
    local model_name=$2
    : ${percentile:?} ${window_size:?} ${batch_size:?}
    local log_dir=${log_dir:=logs/icSHAPE}
    if [ -n "$dense" ];then
        local dense_str=".dense"
    fi

    echo "train: percentile=$percentile, window_size=$window_size, $data_name, $model_name"
    make_dir "$log_dir"
    make_dir trained_models/icSHAPE
    bin/deepfold2.py train_RNAfold --model models/RNAfold/${model_name}.py --batch-size $batch_size \
        --window-size $window_size \
        -i data/icSHAPE/${data_name}.p${percentile}.w${window_size}${dense_str}.train.h5 \
        --save-model trained_models/icSHAPE/${data_name}.p${percentile}.w${window_size}.${model_name}.h5 \
        > logs/icSHAPE/train.${data_name}.p${percentile}.w${window_size}.${model_name}.log 2>&1

}

evaluate_icSHAPE(){
    local data_name=$1
    local model_name=$2
    : ${percentile:?} ${window_size:?} ${batch_size:?}
    local log_dir=${log_dir:=logs/icSHAPE}
    if [ -n "$dense" ];then
        local dense_str=".dense"
    fi

    echo "evaluate: percentile=$percentile, window_size=$window_size, $data_name, $model_name"
    make_dir "$log_dir"
    bin/deepfold2.py test_RNAfold --load-model trained_models/icSHAPE/${data_name}.p${percentile}.w${window_size}.${model_name}.h5 \
        --batch-size $batch_size \
        -i data/icSHAPE/${data_name}.p${percentile}.w${window_size}${dense_str}.test.h5 \
        > logs/icSHAPE/test.${data_name}.p${percentile}.w${window_size}.${model_name}.log 2>&1
}

get_metrics_icSHAPE(){
  local data_name=$1
  local model_name=$2
  : ${percentile:?} ${window_size:?}
  local acc=$(awk "{if(match(\$0, /\['loss', '[a-z_]+'\] = \[(.+), (.+)\]/, a) > 0) print a[2]}" \
    logs/icSHAPE/test.${data_name}.p${percentile}.w${window_size}.${model_name}.log)
  echo -e "$data_name\t$percentile\t$window_size\t$model_name\t$acc"
}


foreach(){
    python - "Datasets_InVitro,Datasets_InVivo" "5-95,10-90" "51,101" "basic,conv1,fcn1,fcn2" <<PYTHON
from itertools import product
import sys
for items in product(map(str.split, sys.argv[1:])):
    print '\t'.join(items)
PYTHON
}

foreach_icSHAPE(){
    for data_name in Datasets_InVitro Datasets_InVivo;do
        for percentile in 5-95 10-90;do
            for window_size in 51 101;do
                for model_name in basic conv1 fcn1 fcn2;do
                    percentile=$percentile window_size=$window_size bin/pipelines.sh train_icSHAPE $data_name $model_name
                done
            done
        done
    done
}

_func_list_after=$(declare -F | cut -d' ' -f3)
command_list="$(echo "$_func_list_before"$'\n'"$_func_list_after" | sort | uniq -u)"

if [ "$#" -lt 1 ];then
  echo "Error: missing arguments"
  echo "Usage: $0 command [options]"
  echo "Available commands: $command_list"
  exit 1
fi

cmd=$1
shift 1
$cmd $@
