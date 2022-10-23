
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

cd ${HOME}/Transformers/IvonNER/IVON_lang_tasks/t5x_Jax


export GOOGLE_CLOUD_BUCKET_NAME=t5xtinetuning
export TFDS_DATA_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/data
export MODEL_DIR=gs://$GOOGLE_CLOUD_BUCKET_NAME/t5x/$(date +%Y%m%d)

# Pre-download dataset in multi-host experiments.
tfds build wmt_t2t_translate --data_dir=$TFDS_DATA_DIR

git clone https://github.com/google-research/t5x
cd ./t5x/

python3 ./t5x/scripts/xm_launch.py \
  --gin_file=t5x/examples/t5/t5_1_1/examples/base_wmt_from_scratch.gin \
  --model_dir=$MODEL_DIR \
  --tfds_data_dir=$TFDS_DATA_DIR