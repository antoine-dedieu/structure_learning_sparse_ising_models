#!/bin/bash
dataset_root='/tmp'

script_name='compare_ising_models.py'

P=16
rho=0.5
type='periodic_graph_uniform_sign'
python $script_name 0 $dataset_root 500 $P $rho $type &
python $script_name 1 $dataset_root 500 $P $rho $type &
python $script_name 2 $dataset_root 500 $P $rho $type &
python $script_name 3 $dataset_root 500 $P $rho $type &
wait
python $script_name 0 $dataset_root 1000 $P $rho $type &
python $script_name 1 $dataset_root 1000 $P $rho $type &
python $script_name 2 $dataset_root 1000 $P $rho $type &
python $script_name 3 $dataset_root 1000 $P $rho $type &
wait
python $script_name 0 $dataset_root 1500 $P $rho $type &
python $script_name 1 $dataset_root 1500 $P $rho $type &
python $script_name 2 $dataset_root 1500 $P $rho $type &
python $script_name 3 $dataset_root 1500 $P $rho $type &
wait
python $script_name 0 $dataset_root 2000 $P $rho $type &
python $script_name 1 $dataset_root 2000 $P $rho $type &
python $script_name 2 $dataset_root 2000 $P $rho $type &
python $script_name 3 $dataset_root 2000 $P $rho $type &

