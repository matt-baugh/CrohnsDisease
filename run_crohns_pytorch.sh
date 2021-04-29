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
  -record_shape 37,99,99 \
  -feature_shape 31,87,87 \
  -gpus 0,1 \
  -axt2=1\
  -cort2=0\
  -axpc=0\
  -at=1 \
  -f=${fold} \
  -bS=32 \
  -lD=CrohnsDisease/log_attention/${TIMESTAMP}fold${fold}/ \
  -nB=20 \
  -mode="train" \
  -mP="CrohnsDisease/trained_models/best_model/fold${fold}"
done
