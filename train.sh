BUCKET_NAME="gs://entails"
JOB_NAME="pwn256"
gcloud ml-engine jobs submit training "${JOB_NAME}" \
--stream-logs \
--module-name entailment.train_possible_worlds_net \
--package-path entailment \
--staging-bucket "${BUCKET_NAME}" \
--region "us-central1" \
--runtime-version=1.8 \
--scale-tier="basic-gpu" \
-- \
--batch_size=64 \
--logdir=$BUCKET_NAME/$JOB_NAME/logs/0 \
--datadir=$BUCKET_NAME/entailment/logical_entailment_dataset/data \
--n_world=256
