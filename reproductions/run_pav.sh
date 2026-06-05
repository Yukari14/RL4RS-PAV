#!/bin/bash

conda activate rl4rs
script_abs=$(readlink -f "$0")
rl4rs_benchmark_dir=$(dirname $script_abs)/..
rl4rs_output_dir=${rl4rs_benchmark_dir}/output
rl4rs_dataset_dir=${rl4rs_benchmark_dir}/dataset
script_dir=${rl4rs_benchmark_dir}/script
export rl4rs_benchmark_dir && export rl4rs_output_dir && export rl4rs_dataset_dir

algo=${1:-CQL}
env_name=${2:-SlateRecEnv-v0}
trial_name=${3:-a_all}

cd ${script_dir}

if [ "${env_name}" = "SeqSlateRecEnv-v0" ]; then
  sample_file="${rl4rs_dataset_dir}/rl4rs_dataset_b3_shuf.csv"
  model_file="${rl4rs_output_dir}/simulator_b2_dien/model"
else
  sample_file="${rl4rs_dataset_dir}/rl4rs_dataset_a_shuf.csv"
  model_file="${rl4rs_output_dir}/simulator_a_dien/model"
fi

common_config="{'env':'${env_name}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${sample_file}','model_file':'${model_file}','trial_name':'${trial_name}'}"

python -u batchrl_train.py ${algo} 'dataset_generate' "${common_config}" &&
python -u pav_train.py 'shape_dataset' "${common_config}" &&
python -u pav_train.py 'diagnostics' "${common_config}" &&
python -u batchrl_train.py ${algo} 'train' "{'env':'${env_name}','trial_name':'${trial_name}','use_pav':True}" &&
python -u batchrl_train.py ${algo} 'eval' "{'env':'${env_name}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${sample_file}','model_file':'${model_file}','trial_name':'${trial_name}','use_pav':True,'gpu':False}" &&
python -u batchrl_train.py ${algo} 'ope' "{'env':'${env_name}','iteminfo_file':'${rl4rs_dataset_dir}/item_info.csv','sample_file':'${sample_file}','model_file':'${model_file}','trial_name':'${trial_name}','use_pav':True,'gpu':False}"
