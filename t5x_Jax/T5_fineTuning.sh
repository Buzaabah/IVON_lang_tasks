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
## cd $HOME/repos/iVON/training_scripts



# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
MODEL_DIR="..."

# Data dir to save the processed dataset in "gs://data_dir" format.
TFDS_DATA_DIR="..."
T5X_DIR="..."  # directory where the T5X repo is cloned.

#python3 ${T5X_DIR}/Transformers/IvonNER/IVON_lang_tasks/t5x_Jax/t5_models/train.py \
cd ${HOME}/Transformers/IvonNER/IVON_lang_tasks/t5x_Jax/t5_models
python3 train.py \
  --gin_file="t5x/examples/t5/t5_1_1/examples/base_wmt_finetune.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}