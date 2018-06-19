BUCKET_NAME="gs://entails"
gcloud ml-engine jobs submit training "entail_test11" \
--stream-logs \
--module-name entailment.train_possible_worlds_net \
--package-path entailment \
--staging-bucket "${BUCKET_NAME}" \
--region "us-central1" \
--runtime-version=1.8 \
-- \
--batch_size=50 \
--logdir="$BUCKET_NAME/logs/0" \
--datadir="$BUCKET_NAME/entailment/logical_entailment_dataset/data"
