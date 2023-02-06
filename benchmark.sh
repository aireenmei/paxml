gcloud alpha compute tpus tpu-vm ssh aireenmei-v4 --zone=us-central2-b --worker=all \
--batch-size=32 \
--command="gsutil cp gs://pax-on-cloud-tpu-project/aireenmei/wheels/20221212/praxis-*.whl ~/ && \
gsutil cp gs://pax-on-cloud-tpu-project/aireenmei/wheels/20221212/paxml-*.whl ~/ && \
pip install ~/praxis-*.whl && pip install ~/paxml-*.whl && \
pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && \
pip uninstall -y jax && \
pip install 'jax @ git+https://github.com/google/jax' && \
pip uninstall -y tensorboard-plugin-wit && \
pip install tbp-nightly"

#pip install 'jax @ git+https://github.com/google/jax' && \

#copy optimizer_prefix_vectorization.py
gcloud alpha compute tpus tpu-vm scp \
~/praxis-aireenmei/praxis/praxis/optimizer_prefix_vectorization.py \
aireenmei-v4:/home/aireenmei/.local/lib/python3.8/site-packages/praxis/optimizer_prefix_vectorization.py \
--zone=us-central2-b --worker=all

#copy paxml/$SCP_FILENAME
export TPU_NAME=aireenmei-v4
export SCP_FILENAME="main.py" && \
export SCP_FILENAME="tasks/lm/params/c4_benchmark.py" && \
gcloud alpha compute tpus tpu-vm scp \
~/paxml-aireenmei/paxml/paxml/$SCP_FILENAME \
$TPU_NAME:/home/aireenmei/.local/lib/python3.8/site-packages/paxml/$SCP_FILENAME \
--zone=us-central2-b --worker=all

# copy c4.py
gcloud alpha compute tpus tpu-vm scp \
~/paxml-aireenmei/paxml/paxml/tasks/lm/params/c4.py \
aireenmei-v3:/home/aireenmei/.local/lib/python3.8/site-packages/paxml/tasks/lm/params/c4_benchmark.py \
--zone=us-central2-b --worker=all --batch-size=32

# copy lm_cloud.py
gcloud alpha compute tpus tpu-vm scp \
~/paxml-aireenmei/paxml/paxml/tasks/lm/params/lm_cloud.py \
aireenmei-v3:/home/aireenmei/.local/lib/python3.8/site-packages/paxml/tasks/lm/params/lm_cloud.py \
--zone=us-central2-b --worker=all

# copy trainer_lib.py
gcloud alpha compute tpus tpu-vm scp \
~/paxml-aireenmei/paxml/paxml/trainer_lib.py \
aireenmei-v4:/home/aireenmei/.local/lib/python3.8/site-packages/paxml/trainer_lib.py \
--zone=us-central2-b --worker=all

#Synthetic data
gcloud alpha compute tpus tpu-vm ssh aireenmei-v3 --zone=us-central2-b --worker=all \
--batch-size=4 \
--command="screen -S paxml -dm bash -c 'python3 /home/aireenmei/.local/lib/python3.8/site-packages/paxml/main.py \
--exp=tasks.lm.params.lm_cloud.LmCloudSpmd16B \
--job_log_dir=gs://aireenmei-us-central1/paxml-gpt-1208-2022/spmd16B-batch512-2 \
--enable_checkpoint_saving=False \
--jax_profiler_port 6000 2>&1 \
| tee ~/log_1208_spmd16B-batch512-2.txt'"
# --job_log_dir=gs://aireenmei-us-central1/paxml-gpt-1207-2022/Spmd16B-syn-batch512 \

# C4 data
gcloud alpha compute tpus tpu-vm ssh aireenmei-v4 --zone=us-central2-b --worker=all \
--batch-size=32 \
--command="screen -S paxml -dm bash -c \
'python3 /home/aireenmei/.local/lib/python3.8/site-packages/paxml/main.py \
--exp=tasks.lm.params.c4_benchmark.C4Spmd256BAdam512Replicas \
--job_log_dir=gs://aireenmei-us-central1/paxml-gpt-1209-2022/C4_256B-batch2048-2 \
--enable_checkpoint_saving=False \
--jax_profiler_port 7000 2>&1 \
| tee ~/log_1209_C4_256B-batch2048-2.txt'"

# C4 data
gcloud alpha compute tpus tpu-vm ssh aireenmei-v4 --zone=us-central2-b --worker=all \
--command="python3 /home/aireenmei/.local/lib/python3.8/site-packages/paxml/main.py \
--exp=tasks.lm.params.c4.C4SpmdGpt3L16AdamOrgHP32Replicas \
--job_log_dir=gs://aireenmei-us-central1/paxml-gpt-1208-2022/C4Spmd_32R-1 \
--enable_checkpoint_saving=False \
--jax_profiler_port 7000 2>&1 \
| tee ~/log_1207_Gpt3L16_32R-1.txt"

#profile
# aireenmei-v3
gcloud compute tpus tpu-vm ssh aireenmei-v3 --worker=0 --zone=us-central2-b \
--ssh-flag="-4 -L 9001:localhost:9001"

# aireenmei-v4
gcloud compute tpus tpu-vm ssh aireenmei-v4 --worker=127 --zone=us-central2-b \
--ssh-flag="-4 -L 9009:localhost:9009"

gcloud alpha compute tpus tpu-vm ssh aireenmei-v4 --worker=all --zone=us-central2-b \
--batch-size=16 \
--command="python3 -c 'import jax; print(jax.device_count())' &"
# tensorboard
TPU_LOAD_LIBRARY=0 tensorboard --logdir ~/tensorboard --port 9001

# Screen 
screen -ls
screen -r <id>