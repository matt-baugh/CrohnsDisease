FOLDS=$1

BASE="/vol/bitbucket/mb4617"
RECORDS="MRI_Crohns/numpy_datasets/ti_imb/axial_t2_only"
# RECORDS="MRI_Crohns/numpy_datasets/ti_imb_generic/axial_t2_only"
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
  -py=true \
  -at=1 \
  -f=${fold} \
  -bS=16 \
  -lD=CrohnsDisease/log_attention/${TIMESTAMP}/ \
  -nB=1200 \
  -mode="train" \
  -mP="CrohnsDisease/trained_models/best_model/fold${fold}"
done
