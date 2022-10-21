#$ -jc gpu-container_g1
#$ -ac d=nvcr-tensorflow-2105-tf2-py3
. /fefs/opt/dgx/env_set/nvcr-tensorflow-2105-tf2-py3.sh
# Setting/unsetting proxy for internet access
proxy_pcc() {
    export MY_PROXY_URL=http://10.1.30.1:8080/
    export HTTP_PROXY=$MY_PROXY_URL
    export HTTPS_PROXY=$MY_PROXY_URL
    export FTP_PROXY=$MY_PROXY_URL
    export http_proxy=$MY_PROXY_URL
    export https_proxy=$MY_PROXY_URL
    export ftp_proxy=$MY_PROXY_URL
}
proxy_gpu() {
    export MY_PROXY_URL="http://10.1.10.1:8080/"
    export HTTP_PROXY=$MY_PROXY_URL
    export HTTPS_PROXY=$MY_PROXY_URL
    export FTP_PROXY=$MY_PROXY_URL
    export http_proxy=$MY_PROXY_URL
    export https_proxy=$MY_PROXY_URL
    export ftp_proxy=$MY_PROXY_URL
}
proxy_clear() {
    unset MY_PROXY_URL
    unset HTTP_PROXY
    unset HTTPS_PROXY
    unset ftp_proxy
    unset http_proxy
    unset https_proxy
    unset ftp_proxy
}
proxy_clear
proxy_gpu

### commented out: -pe OpenMP 28
source $HOME/miniconda3/bin/activate base


export TASK_NAME=sstb

cd ${HOME}/Transformers/IvonNER/IVON_lang_tasks/BERT/Text_classificationFlax
python3 run_flax_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name ${TASK_NAME} \
  --max_seq_length 128 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 4 \
  --eval_steps 100 \
  --output_dir ./$TASK_NAME/ \
  #--push_to_hub