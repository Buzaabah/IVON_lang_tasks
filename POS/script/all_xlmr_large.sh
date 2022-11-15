#$ -jc gpu-container_g1
#$ -cwd
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
source $HOME/miniconda3/bin/activate Torch_env

for LANG in bam bbj ewe fon hau ibo kin lug luo nya pcm sna swa twi wol xho yor zul
do
	for j in 1 2 3 4 5
	do
		export MAX_LENGTH=200
		export BERT_MODEL=xlm-roberta-large
		export OUTPUT_DIR=baseline_models/${LANG}_xlmrlarge
		export TEXT_RESULT=test_result$j.txt
		export TEXT_PREDICTION=test_predictions$j.txt
		export BATCH_SIZE=16
		export NUM_EPOCHS=20
		export SAVE_STEPS=10000
		export SEED=$j

		CUDA_VISIBLE_DEVICES=3 python ../train_pos.py --data_dir ../data/${LANG}/ \
		--model_type xlmroberta \
		--model_name_or_path $BERT_MODEL \
		--output_dir $OUTPUT_DIR \
		--test_result_file $TEXT_RESULT \
		--test_prediction_file $TEXT_PREDICTION \
		--max_seq_length  $MAX_LENGTH \
		--num_train_epochs $NUM_EPOCHS \
		--per_gpu_train_batch_size $BATCH_SIZE \
		--save_steps $SAVE_STEPS \
		--gradient_accumulation_steps 2 \
		--seed $SEED \
		--do_train \
		--do_eval \
		--do_predict \
		--overwrite_output_dir
	done
done
