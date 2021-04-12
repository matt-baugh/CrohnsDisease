

BASE="/vol/bitbucket/mb4617"
LOCALISED_DATAPATH="MRI_Crohns/numpy_datasets/ti_imb/all_data"
GENERALISED_DATAPATH="MRI_Crohns/numpy_datasets/ti_imb_generic/all_data"

multimodal_settings=(0 1)
attention_settings=(0)
localisation_settings=(${LOCALISED_DATAPATH})
folds=(0)

for mode in ${multimodal_settings[@]}
  do
    for loc in ${localisation_settings[@]}
      do
        if [[ ${loc} == ${LOCALISED_DATAPATH} ]]; then
          loc_i=1
        else
          loc_i=0
        fi
        for att in ${attention_settings[@]}
          do
            TIMESTAMP=`date +%Y-%m-%d_%H:%M:%S`
            for fold in ${folds[@]}
            do

              model_dir="CrohnsDisease/trained_models/2/original_dataset_mode${mode}loc${loc_i}att${att}"
              mkdir -p "${BASE}/${model_dir}"

              python3 run_pytorch.py \
                Crohns_MRI \
                ${BASE} \
                ${loc}_train_fold${fold}.npz \
                ${loc}_test_fold${fold}.npz \
                -record_shape 99,99,99 \
                -feature_shape 87,87,87 \
                -gpus 1 \
                -py=true \
                -axt2=1\
                -cort2=${mode}\
                -axpc=${mode}\
                -at=${att} \
                -f=${fold} \
                -bS=48 \
                -lD=CrohnsDisease/log_second_round/mode${mode}loc${loc_i}att${att}${TIMESTAMP}fold${fold}/ \
                -nB=20 \
                -mode="train" \
                -mP="${model_dir}/fold${fold}"
            done
          done
      done
  done