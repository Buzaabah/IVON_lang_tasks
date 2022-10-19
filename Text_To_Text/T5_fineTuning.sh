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

cd ${HOME}/Transformers/IvonNER/IVON_lang_tasks/Text_To_Text
t5_mesh_transformer  \
  --model_dir="${MODEL_DIR}" \
  --t5_tfds_data_dir="${DATA_DIR}" \
  --gin_file="dataset.gin" \
  --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:0']" \
  --gin_param="MIXTURE_NAME = 'glue_mrpc_v002'" \
  --gin_file="gs://t5-data/pretrained_models/small/operative_config.gin"

