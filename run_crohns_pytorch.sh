FOLDS=$1

BASE="/vol/bitbucket/mb4617"
RECORDS="MRI_Crohns/numpy_datasets/ti_imb/all_data"
# RECORDS="MRI_Crohns/numpy_datasets/ti_imb_generic/all_data"
TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`


echo "Running ${#@} fold(s)"

for fold in ${@}
do
  python3 run_pytorch.py \
  Crohns_MRI \
  ${BASE} \
  ${RECORDS}_train_fold${fold}.npz \
  ${RECORDS}_test_fold${fold}.npz \
  -record_shape 99,99,99 \
  -feature_shape 87,87,87 \
  -gpus 0 \
  -py=true \
  -axt2=1\
  -cort2=1\
  -axpc=1\
  -at=1 \
  -f=${fold} \
  -bS=64 \
  -lD=CrohnsDisease/log_attention/${TIMESTAMP}fold${fold}/ \
  -nB=1200 \
  -mode="train" \
  -mP="CrohnsDisease/trained_models/best_model/fold${fold}"
done
