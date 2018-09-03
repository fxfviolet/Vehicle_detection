#!/bin/bash
# 查找脚本所在路径，并进入
#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

# 定义各目录
output_dir=/output  # 训练目录
dataset_dir=/data/fxfviolet/vehicle_train   

train_dir=$output_dir/train
checkpoint_dir=$train_dir
eval_dir=$output_dir/eval

# config文件
config=ssd_mobilenet.config
pipeline_config_path=$output_dir/$config

rm -rvf $output_dir/*

cp $DIR/$config $pipeline_config_path

for i in {0..4}   
do
    echo "############" $i "runnning #################"
    last=$[$i*100]
    current=$[($i+1)*100]
    sed -i "s/^  num_steps: $last$/  num_steps: $current/g" $pipeline_config_path  

    echo "############" $i "training #################"
    python ./object_detection/train.py --train_dir=$train_dir --pipeline_config_path=$pipeline_config_path

    echo "############" $i "evaluating, this takes a long while #################"
    python ./object_detection/eval.py --checkpoint_dir=$checkpoint_dir --eval_dir=$eval_dir --pipeline_config_path=$pipeline_config_path
done

# 导出模型
python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path $pipeline_config_path --trained_checkpoint_prefix $train_dir/model.ckpt-$current  --output_directory $output_dir/exported_graphs

# 在test.jpg上验证导出的模型
python ./inference.py --output_dir=$output_dir --dataset_dir=$dataset_dir
