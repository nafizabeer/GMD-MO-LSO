# Meta flags
gpu="--gpu"  # change to "" if no GPU is to be used
seed_array=( 1)
root_dir="logs/opt_reduced_dataset/top20_NP_score_DRD2"
# start_model="assets/pretrained_models/chem.ckpt"
start_model="assets/pretrained_models/top20_NP_score_DRD2/epoch=29.ckpt"
query_budget=500
n_retrain_epochs=1
n_init_retrain_epochs=1

# Experiment : weighted retraining with different parameters
# ==================================================
k_high="1e-1"
k_low="1e-2"
# k_low2="1e-4"
r_high=100
r_low=50
r_inf="1000000"  # Set to essentially be infinite (since "inf" is not supported)
weight_type="rank_pf" 
lso_strategy="sample" #"opt" #

# Set specific experiments to do:

k_expt=(  "$k_low" )
r_expt=(  "$r_low" )

expt_index=0  # Track experiments
for seed in "${seed_array[@]}"; do
    for ((i=0;i<${#k_expt[@]};++i)); do

        # Increment experiment index
        expt_index=$((expt_index+1))

        # Break loop if using slurm and it's not the right task
        if [[ -n "${SLURM_ARRAY_TASK_ID}" ]] && [[ "${SLURM_ARRAY_TASK_ID}" != "$expt_index" ]]
        then
            continue
        fi


        # Echo info of task to be executed
        r="${r_expt[$i]}"
        k="${k_expt[$i]}"
        echo "r=${r} k=${k} seed=${seed}"

        # Run command
        # python weighted_retraining/opt_scripts/opt_chem.py \
        #     --seed="$seed" $gpu \
        #     --query_budget="$query_budget" \
        #     --retraining_frequency="$r" \
        #     --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
        #     --pretrained_model_file="$start_model" \
        #     --lso_strategy="$lso_strategy" \
        #     --train_path="data/chem/zinc/top20_logP_SAS/tensors_train" \
        #     --val_path=data/chem/zinc/orig_model/tensors_val \
        #     --vocab_file=data/chem/zinc/orig_model/vocab.txt \
        #     --property_file=data/chem/zinc/orig_model/logP_all.pkl \
        #     --property2_file=data/chem/zinc/orig_model/sas_all.pkl \
        #     --property='logP' \
        #     --property2='SAS' \
        #     --n_retrain_epochs="$n_retrain_epochs" \
        #     --n_init_retrain_epochs="$n_init_retrain_epochs" \
        #     --n_best_points=1000 --n_rand_points=4000 \
        #     --n_inducing_points=250 \
        #     --weight_type="$weight_type" --rank_weight_k="$k" \
        #     --all_new=1 
        
        # python weighted_retraining/opt_scripts/opt_chem.py \
        #     --seed="$seed" $gpu \
        #     --query_budget="$query_budget" \
        #     --retraining_frequency="$r" \
        #     --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
        #     --pretrained_model_file="$start_model" \
        #     --lso_strategy="$lso_strategy" \
        #     --train_path="data/chem/zinc/top20_logP_NP_score/tensors_train" \
        #     --val_path=data/chem/zinc/top20_logP_NP_score/tensors_val \
        #     --vocab_file=data/chem/zinc/orig_model/vocab.txt \
        #     --property_file=data/chem/zinc/orig_model/logP_all.pkl \
        #     --property2_file=data/chem/zinc/orig_model/NP_score_all.pkl \
        #     --property='logP' \
        #     --property2='NP_score' \
        #     --n_retrain_epochs="$n_retrain_epochs" \
        #     --n_init_retrain_epochs="$n_init_retrain_epochs" \
        #     --n_best_points=2000 --n_rand_points=8000 \
        #     --n_inducing_points=500 \
        #     --weight_type="$weight_type" --rank_weight_k="$k" \
        #     --all_new=1
        
        # python weighted_retraining/opt_scripts/opt_chem.py \
        #     --seed="$seed" $gpu \
        #     --query_budget="$query_budget" \
        #     --retraining_frequency="$r" \
        #     --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
        #     --pretrained_model_file="$start_model" \
        #     --lso_strategy="$lso_strategy" \
        #     --train_path="data/chem/zinc/top20_SAS_NP_score/tensors_train" \
        #     --val_path=data/chem/zinc/top20_SAS_NP_score/tensors_val \
        #     --vocab_file=data/chem/zinc/top20_SAS_NP_score/vocab.txt \
        #     --property_file=data/chem/zinc/orig_model/NP_score_all.pkl\
        #     --property2_file=data/chem/zinc/orig_model/sas_all.pkl \
        #     --property='NP_score' \
        #     --property2='SAS'\
        #     --n_retrain_epochs="$n_retrain_epochs" \
        #     --n_init_retrain_epochs="$n_init_retrain_epochs" \
        #     --n_best_points=2000 --n_rand_points=8000 \
        #     --n_inducing_points=500 \
        #     --weight_type="$weight_type" --rank_weight_k="$k" \
        #     --all_new=1
        
        # python weighted_retraining/opt_scripts/opt_chem.py \
        #     --seed="$seed" $gpu \
        #     --query_budget="$query_budget" \
        #     --retraining_frequency="$r" \
        #     --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
        #     --pretrained_model_file="$start_model" \
        #     --lso_strategy="$lso_strategy" \
        #     --train_path="data/chem/zinc/top20_logP_DRD2/tensors_train" \
        #     --val_path=data/chem/zinc/orig_model/tensors_val \
        #     --vocab_file=data/chem/zinc/orig_model/vocab.txt \
        #     --property_file=data/chem/zinc/orig_model/logP_all.pkl \
        #     --property2_file=data/chem/zinc/orig_model/DRD2_all.pkl \
        #     --property='logP' \
        #     --property2='DRD2' \
        #     --n_retrain_epochs="$n_retrain_epochs" \
        #     --n_init_retrain_epochs="$n_init_retrain_epochs" \
        #     --n_best_points=2000 --n_rand_points=8000 \
        #     --n_inducing_points=500 \
        #     --weight_type="$weight_type" --rank_weight_k="$k" \
        #     --all_new=1
        
        # python weighted_retraining/opt_scripts/opt_chem.py \
        #     --seed="$seed" $gpu \
        #     --query_budget="$query_budget" \
        #     --retraining_frequency="$r" \
        #     --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
        #     --pretrained_model_file="$start_model" \
        #     --lso_strategy="$lso_strategy" \
        #     --train_path="data/chem/zinc/top20_SAS_DRD2/tensors_train" \
        #     --val_path=data/chem/zinc/top20_SAS_DRD2/tensors_val \
        #     --vocab_file=data/chem/zinc/top20_SAS_DRD2/vocab.txt \
        #     --property_file=data/chem/zinc/orig_model/DRD2_all.pkl \
        #     --property2_file=data/chem/zinc/orig_model/sas_all.pkl \
        #     --property='DRD2' \
        #     --property2='SAS' \
        #     --n_retrain_epochs="$n_retrain_epochs" \
        #     --n_init_retrain_epochs="$n_init_retrain_epochs" \
        #     --n_best_points=2000 --n_rand_points=8000 \
        #     --n_inducing_points=500 \
        #     --weight_type="$weight_type" --rank_weight_k="$k" \
        #     --all_new=1
        
        python weighted_retraining/opt_scripts/opt_chem.py \
            --seed="$seed" $gpu \
            --query_budget="$query_budget" \
            --retraining_frequency="$r" \
            --result_root="${root_dir}/${weight_type}/k_${k}/r_${r}/seed${seed}" \
            --pretrained_model_file="$start_model" \
            --lso_strategy="$lso_strategy" \
            --train_path="data/chem/zinc/top20_NP_score_DRD2/tensors_train" \
            --val_path=data/chem/zinc/top20_NP_score_DRD2/tensors_val \
            --vocab_file=data/chem/zinc/orig_model/vocab.txt \
            --property_file=data/chem/zinc/orig_model/NP_score_all.pkl\
            --property2_file=data/chem/zinc/orig_model/DRD2_all.pkl \
            --property='NP_score' \
            --property2='DRD2'\
            --n_retrain_epochs="$n_retrain_epochs" \
            --n_init_retrain_epochs="$n_init_retrain_epochs" \
            --n_best_points=2000 --n_rand_points=8000 \
            --n_inducing_points=500 \
            --weight_type="$weight_type" --rank_weight_k="$k" \
            --all_new=1

    done
done
