# BioNER with XLNet-CRF

This repository provides the code for fine-tuning Shared Labels and Dynamic Splicing model on XLNet-CRF network backbone.
two noise reduction learning approach for biomedical named entity recognition.

## Download

You can download all datasets we used at https://github.com/cambridgeltl/MTL-Bioinformatics-2016. Pre-training models have been made available at https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip

Considering that the training checkpoint is too big(over 2TB), in the future we will choose some of them to upload.

## Installation

This section describe the installation on Tensorflow 1.X. 
We highly recommend using GPU with more than 16GB memory and tensorFlow GPU support requires can be found at https://www.tensorflow.org/install/gpu and https://www.tensorflow.org/install/source#gpu.

We only test Tensorflow 1.14 and 1.15 and version 1.15.2 is recommended. all requirements can be seen at requrements.txt

    pip install -r requirements.txt

## Preprocessing

Datasets preprocessing code is in create_data.py. In order to execute comparison experiments and ablation studies we generated 3 types of label: label_id, label_x_id and label_gather and 2 types of label mask: label_mask_x and label_mask_gather.
Label_index is used to record the position of each label.
Here is an example of them:
* sentence: selegiline-induced postural hypotension in parkinson's disease ...
* piece:  __s ele gi line - __induced __post ural __hypo tension __in __park in son __' __s __disease ...
* tokens: 17, 23, 6159, 3141, 814, 17, 13, 17, 12674, 701, 9323, 11581, 23157, 25, 2133, 153, 672, 17, 26, 17, 23, 1487, 17 ...
* label_id: 4, 4, 4, 4, 4, 0, 0, 0, 0, 5, 5, 7, 7, 0, 5, 5, 5, 6, 6, 6, 6, 7, 0 ...
* label_x_id: 4, 9, 9, 9, 9, 0, 9, 0, 9, 5, 9, 7, 9, 0, 5, 9, 9, 6, 9, 6, 9, 7, 0 ...
* label_gather: 4, 0, 0, 5, 7, 0, 5, 6, 6, 7, 0 ...
* label_mask_x: 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1 ...
* label_mask_gather: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ...
* label_index: 0, 5, 7, 9, 11, 13, 14, 17, 19, 21, 22 ...

## Fine-tuning

You can do MTL-XC by:

    python run_crf_multi_task.py --use_tpu=False \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --model_config_path=xlnet_cased_L-24_H-1024_A-16/xlnet_config.json \
    --spiece_model_file=xlnet_cased_L-24_H-1024_A-16/spiece.model \
    --init_checkpoint=xlnet_cased_L-24_H-1024_A-16/xlnet_model.ckpt \
    --model_dir=model_large \
    --lower=True \
    --max_seq_length=512 \
    --do_train=True \
    --train_batch_size=6 \
    --do_eval=True \
    --do_predict=true \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --adam_epsilon=1e-6 \
    --save_steps=500 \
    --max_save=5 \
    --warmup_steps=0 \
    --cache_dir=cache_${task} \
    --predict_dir=predict_${task} \
    --label_mode=gather \
    --label_mask=gather \
    --task=${task} \
    --no_crf=False
    
and MTL-LS by:

    python run_crf.py --use_tpu=False \
    --num_hosts=1 \
    --num_core_per_host=1 \
    --model_config_path=${model}/xlnet_config.json \
    --spiece_model_file=${model}/spiece.model \
    --init_checkpoint=${model}/xlnet_model.ckpt \
    --model_dir=${model_name} \
    --lower=True \
    --max_seq_length=512 \
    --do_train=True \
    --train_batch_size=6 \
    --do_eval=False \
    --do_predict=true \
    --eval_batch_size=32 \
    --learning_rate=2e-5 \
    --adam_epsilon=1e-6 \
    --save_steps=300 \
    --max_save=5 \
    --train_steps=${step} \
    --warmup_steps=0 \
    --cache_dir=cache_chemical \
    --predict_dir=predict_chem_bc4 \
    --label_mode=gather \
    --label_mask=gather \
    --task=${task} \
    --no_crf=False \
    --init_mtl_checkpoint=model_large_chemical_crf_gather_gather/model.ckpt-74500 \
    --mtl_start=0 \
    --mtl_end=317 \
    --crf_classes=5

You need to replace `${}` as your Configuration.
It can also pre-training or fine-tuning on multi-GPU cards by using `adamw_optimizer.py` we provided.
The mtl_end is the slicing point.

## Result

You can find all result including evaluated and prediction at `predict_dir`.
Evaluated will be like this:

    Eval result : eval_accuracy 0.9896355867385864 
        eval_loss 33.814178466796875 
        f1 0.9896355271339417 
        global_step 4500 
        loss 33.814178466796875 
        path model_large_anatem_crf_gather_gather/model.ckpt-4500 
        step 4500 
Prediction result will be like:

    2400
    processed 99976 tokens with 4616 phrases; found: 4567 phrases; correct: 4146.
    accuracy:  98.85%; precision:  90.78%; recall:  89.82%; FB1:  90.30
              Anatomy: precision:  90.78%; recall:  89.82%; FB1:  90.30  4567
              
## License and Disclaimer

Please see the LICENSE file for details. Downloading data indicates your acceptance of our disclaimer.