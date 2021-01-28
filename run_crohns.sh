FOLDS=$1

BASE="/vol/bitbucket/rh2515"
RECORDS="MRI_Crohns/tfrecords/ti_imb/axial_t2_only"
# RECORDS="MRI_Crohns/tfrecords/ti_imb_generic/axial_t2_only"
TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`


echo "Running ${#@} fold(s)"

for fold in ${@}
do
  python3 run.py \
  Crohns_MRI \
  ${BASE} \
  ${RECORDS}_train_fold${fold}.tfrecords \
  ${RECORDS}_test_fold${fold}.tfrecords \
  -record_shape 37,99,99 \
  -feature_shape 31,87,87 \
  -at=1 \
  -f=${fold} \
  -bS=64 \
  -lD=CrohnsDisease/log_attention/${TIMESTAMP}/ \
  -nB=1200 \
  -mode="test" \
  -mP="CrohnsDisease/trained_models/best_model/fold${fold}"
done
