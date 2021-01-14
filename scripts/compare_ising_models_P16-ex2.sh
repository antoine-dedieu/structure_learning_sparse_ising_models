#!/bin/bash
dataset_root='/tmp'

script_name='compare_ising_models.py'

P=16
rho=0.7
type='random_graph_uniform_sign_uniform_val'
python $script_name 0 $dataset_root 500 $P $rho $type &
python $script_name 1 $dataset_root 500 $P $rho $type &
python $script_name 2 $dataset_root 500 $P $rho $type &
python $script_name 3 $dataset_root 500 $P $rho $type &
python $script_name 4 $dataset_root 500 $P $rho $type &
python $script_name 5 $dataset_root 500 $P $rho $type &
python $script_name 6 $dataset_root 500 $P $rho $type &
python $script_name 7 $dataset_root 500 $P $rho $type &
python $script_name 8 $dataset_root 500 $P $rho $type &
python $script_name 9 $dataset_root 500 $P $rho $type &
python $script_name 10 $dataset_root 500 $P $rho $type &
python $script_name 11 $dataset_root 500 $P $rho $type &
python $script_name 12 $dataset_root 500 $P $rho $type &
python $script_name 13 $dataset_root 500 $P $rho $type &
python $script_name 14 $dataset_root 500 $P $rho $type &
python $script_name 15 $dataset_root 500 $P $rho $type &
python $script_name 16 $dataset_root 500 $P $rho $type &
python $script_name 17 $dataset_root 500 $P $rho $type &
python $script_name 18 $dataset_root 500 $P $rho $type &
python $script_name 19 $dataset_root 500 $P $rho $type &
python $script_name 20 $dataset_root 500 $P $rho $type &
python $script_name 21 $dataset_root 500 $P $rho $type &
python $script_name 22 $dataset_root 500 $P $rho $type &
python $script_name 23 $dataset_root 500 $P $rho $type &
python $script_name 24 $dataset_root 500 $P $rho $type &
python $script_name 25 $dataset_root 500 $P $rho $type &
python $script_name 26 $dataset_root 500 $P $rho $type &
python $script_name 27 $dataset_root 500 $P $rho $type &
python $script_name 28 $dataset_root 500 $P $rho $type &
python $script_name 29 $dataset_root 500 $P $rho $type &
wait
python $script_name 0 $dataset_root 1000 $P $rho $type &
python $script_name 1 $dataset_root 1000 $P $rho $type &
python $script_name 2 $dataset_root 1000 $P $rho $type &
python $script_name 3 $dataset_root 1000 $P $rho $type &
python $script_name 4 $dataset_root 1000 $P $rho $type &
python $script_name 5 $dataset_root 1000 $P $rho $type &
python $script_name 6 $dataset_root 1000 $P $rho $type &
python $script_name 7 $dataset_root 1000 $P $rho $type &
python $script_name 8 $dataset_root 1000 $P $rho $type &
python $script_name 9 $dataset_root 1000 $P $rho $type &
python $script_name 10 $dataset_root 1000 $P $rho $type &
python $script_name 11 $dataset_root 1000 $P $rho $type &
python $script_name 12 $dataset_root 1000 $P $rho $type &
python $script_name 13 $dataset_root 1000 $P $rho $type &
python $script_name 14 $dataset_root 1000 $P $rho $type &
python $script_name 15 $dataset_root 1000 $P $rho $type &
python $script_name 16 $dataset_root 1000 $P $rho $type &
python $script_name 17 $dataset_root 1000 $P $rho $type &
python $script_name 18 $dataset_root 1000 $P $rho $type &
python $script_name 19 $dataset_root 1000 $P $rho $type &
python $script_name 20 $dataset_root 1000 $P $rho $type &
python $script_name 21 $dataset_root 1000 $P $rho $type &
python $script_name 22 $dataset_root 1000 $P $rho $type &
python $script_name 23 $dataset_root 1000 $P $rho $type &
python $script_name 24 $dataset_root 1000 $P $rho $type &
python $script_name 25 $dataset_root 1000 $P $rho $type &
python $script_name 26 $dataset_root 1000 $P $rho $type &
python $script_name 27 $dataset_root 1000 $P $rho $type &
python $script_name 28 $dataset_root 1000 $P $rho $type &
python $script_name 29 $dataset_root 1000 $P $rho $type &
wait
python $script_name 0 $dataset_root 1500 $P $rho $type &
python $script_name 1 $dataset_root 1500 $P $rho $type &
python $script_name 2 $dataset_root 1500 $P $rho $type &
python $script_name 3 $dataset_root 1500 $P $rho $type &
python $script_name 4 $dataset_root 1500 $P $rho $type &
python $script_name 5 $dataset_root 1500 $P $rho $type &
python $script_name 6 $dataset_root 1500 $P $rho $type &
python $script_name 7 $dataset_root 1500 $P $rho $type &
python $script_name 8 $dataset_root 1500 $P $rho $type &
python $script_name 9 $dataset_root 1500 $P $rho $type &
python $script_name 10 $dataset_root 1500 $P $rho $type &
python $script_name 11 $dataset_root 1500 $P $rho $type &
python $script_name 12 $dataset_root 1500 $P $rho $type &
python $script_name 13 $dataset_root 1500 $P $rho $type &
python $script_name 14 $dataset_root 1500 $P $rho $type &
python $script_name 15 $dataset_root 1500 $P $rho $type &
python $script_name 16 $dataset_root 1500 $P $rho $type &
python $script_name 17 $dataset_root 1500 $P $rho $type &
python $script_name 18 $dataset_root 1500 $P $rho $type &
python $script_name 19 $dataset_root 1500 $P $rho $type &
python $script_name 20 $dataset_root 1500 $P $rho $type &
python $script_name 21 $dataset_root 1500 $P $rho $type &
python $script_name 22 $dataset_root 1500 $P $rho $type &
python $script_name 23 $dataset_root 1500 $P $rho $type &
python $script_name 24 $dataset_root 1500 $P $rho $type &
python $script_name 25 $dataset_root 1500 $P $rho $type &
python $script_name 26 $dataset_root 1500 $P $rho $type &
python $script_name 27 $dataset_root 1500 $P $rho $type &
python $script_name 28 $dataset_root 1500 $P $rho $type &
python $script_name 29 $dataset_root 1500 $P $rho $type &
wait
python $script_name 0 $dataset_root 2000 $P $rho $type &
python $script_name 1 $dataset_root 2000 $P $rho $type &
python $script_name 2 $dataset_root 2000 $P $rho $type &
python $script_name 3 $dataset_root 2000 $P $rho $type &
python $script_name 4 $dataset_root 2000 $P $rho $type &
python $script_name 5 $dataset_root 2000 $P $rho $type &
python $script_name 6 $dataset_root 2000 $P $rho $type &
python $script_name 7 $dataset_root 2000 $P $rho $type &
python $script_name 8 $dataset_root 2000 $P $rho $type &
python $script_name 9 $dataset_root 2000 $P $rho $type &
python $script_name 10 $dataset_root 2000 $P $rho $type &
python $script_name 11 $dataset_root 2000 $P $rho $type &
python $script_name 12 $dataset_root 2000 $P $rho $type &
python $script_name 13 $dataset_root 2000 $P $rho $type &
python $script_name 14 $dataset_root 2000 $P $rho $type &
python $script_name 15 $dataset_root 2000 $P $rho $type &
python $script_name 16 $dataset_root 2000 $P $rho $type &
python $script_name 17 $dataset_root 2000 $P $rho $type &
python $script_name 18 $dataset_root 2000 $P $rho $type &
python $script_name 19 $dataset_root 2000 $P $rho $type &
python $script_name 20 $dataset_root 2000 $P $rho $type &
python $script_name 21 $dataset_root 2000 $P $rho $type &
python $script_name 22 $dataset_root 2000 $P $rho $type &
python $script_name 23 $dataset_root 2000 $P $rho $type &
python $script_name 24 $dataset_root 2000 $P $rho $type &
python $script_name 25 $dataset_root 2000 $P $rho $type &
python $script_name 26 $dataset_root 2000 $P $rho $type &
python $script_name 27 $dataset_root 2000 $P $rho $type &
python $script_name 28 $dataset_root 2000 $P $rho $type &
python $script_name 29 $dataset_root 2000 $P $rho $type &
wait
python $script_name 0 $dataset_root 2500 $P $rho $type &
python $script_name 1 $dataset_root 2500 $P $rho $type &
python $script_name 2 $dataset_root 2500 $P $rho $type &
python $script_name 3 $dataset_root 2500 $P $rho $type &
python $script_name 4 $dataset_root 2500 $P $rho $type &
python $script_name 5 $dataset_root 2500 $P $rho $type &
python $script_name 6 $dataset_root 2500 $P $rho $type &
python $script_name 7 $dataset_root 2500 $P $rho $type &
python $script_name 8 $dataset_root 2500 $P $rho $type &
python $script_name 9 $dataset_root 2500 $P $rho $type &
python $script_name 10 $dataset_root 2500 $P $rho $type &
python $script_name 11 $dataset_root 2500 $P $rho $type &
python $script_name 12 $dataset_root 2500 $P $rho $type &
python $script_name 13 $dataset_root 2500 $P $rho $type &
python $script_name 14 $dataset_root 2500 $P $rho $type &
python $script_name 15 $dataset_root 2500 $P $rho $type &
python $script_name 16 $dataset_root 2500 $P $rho $type &
python $script_name 17 $dataset_root 2500 $P $rho $type &
python $script_name 18 $dataset_root 2500 $P $rho $type &
python $script_name 19 $dataset_root 2500 $P $rho $type &
python $script_name 20 $dataset_root 2500 $P $rho $type &
python $script_name 21 $dataset_root 2500 $P $rho $type &
python $script_name 22 $dataset_root 2500 $P $rho $type &
python $script_name 23 $dataset_root 2500 $P $rho $type &
python $script_name 24 $dataset_root 2500 $P $rho $type &
python $script_name 25 $dataset_root 2500 $P $rho $type &
python $script_name 26 $dataset_root 2500 $P $rho $type &
python $script_name 27 $dataset_root 2500 $P $rho $type &
python $script_name 28 $dataset_root 2500 $P $rho $type &
python $script_name 29 $dataset_root 2500 $P $rho $type &
wait
python $script_name 0 $dataset_root 3000 $P $rho $type &
python $script_name 1 $dataset_root 3000 $P $rho $type &
python $script_name 2 $dataset_root 3000 $P $rho $type &
python $script_name 3 $dataset_root 3000 $P $rho $type &
python $script_name 4 $dataset_root 3000 $P $rho $type &
python $script_name 5 $dataset_root 3000 $P $rho $type &
python $script_name 6 $dataset_root 3000 $P $rho $type &
python $script_name 7 $dataset_root 3000 $P $rho $type &
python $script_name 8 $dataset_root 3000 $P $rho $type &
python $script_name 9 $dataset_root 3000 $P $rho $type &
python $script_name 10 $dataset_root 3000 $P $rho $type &
python $script_name 11 $dataset_root 3000 $P $rho $type &
python $script_name 12 $dataset_root 3000 $P $rho $type &
python $script_name 13 $dataset_root 3000 $P $rho $type &
python $script_name 14 $dataset_root 3000 $P $rho $type &
python $script_name 15 $dataset_root 3000 $P $rho $type &
python $script_name 16 $dataset_root 3000 $P $rho $type &
python $script_name 17 $dataset_root 3000 $P $rho $type &
python $script_name 18 $dataset_root 3000 $P $rho $type &
python $script_name 19 $dataset_root 3000 $P $rho $type &
python $script_name 20 $dataset_root 3000 $P $rho $type &
python $script_name 21 $dataset_root 3000 $P $rho $type &
python $script_name 22 $dataset_root 3000 $P $rho $type &
python $script_name 23 $dataset_root 3000 $P $rho $type &
python $script_name 24 $dataset_root 3000 $P $rho $type &
python $script_name 25 $dataset_root 3000 $P $rho $type &
python $script_name 26 $dataset_root 3000 $P $rho $type &
python $script_name 27 $dataset_root 3000 $P $rho $type &
python $script_name 28 $dataset_root 3000 $P $rho $type &
python $script_name 29 $dataset_root 3000 $P $rho $type &
wait
python $script_name 0 $dataset_root 3500 $P $rho $type &
python $script_name 1 $dataset_root 3500 $P $rho $type &
python $script_name 2 $dataset_root 3500 $P $rho $type &
python $script_name 3 $dataset_root 3500 $P $rho $type &
python $script_name 4 $dataset_root 3500 $P $rho $type &
python $script_name 5 $dataset_root 3500 $P $rho $type &
python $script_name 6 $dataset_root 3500 $P $rho $type &
python $script_name 7 $dataset_root 3500 $P $rho $type &
python $script_name 8 $dataset_root 3500 $P $rho $type &
python $script_name 9 $dataset_root 3500 $P $rho $type &
python $script_name 10 $dataset_root 3500 $P $rho $type &
python $script_name 11 $dataset_root 3500 $P $rho $type &
python $script_name 12 $dataset_root 3500 $P $rho $type &
python $script_name 13 $dataset_root 3500 $P $rho $type &
python $script_name 14 $dataset_root 3500 $P $rho $type &
python $script_name 15 $dataset_root 3500 $P $rho $type &
python $script_name 16 $dataset_root 3500 $P $rho $type &
python $script_name 17 $dataset_root 3500 $P $rho $type &
python $script_name 18 $dataset_root 3500 $P $rho $type &
python $script_name 19 $dataset_root 3500 $P $rho $type &
python $script_name 20 $dataset_root 3500 $P $rho $type &
python $script_name 21 $dataset_root 3500 $P $rho $type &
python $script_name 22 $dataset_root 3500 $P $rho $type &
python $script_name 23 $dataset_root 3500 $P $rho $type &
python $script_name 24 $dataset_root 3500 $P $rho $type &
python $script_name 25 $dataset_root 3500 $P $rho $type &
python $script_name 26 $dataset_root 3500 $P $rho $type &
python $script_name 27 $dataset_root 3500 $P $rho $type &
python $script_name 28 $dataset_root 3500 $P $rho $type &
python $script_name 29 $dataset_root 3500 $P $rho $type &
wait
python $script_name 0 $dataset_root 4000 $P $rho $type &
python $script_name 1 $dataset_root 4000 $P $rho $type &
python $script_name 2 $dataset_root 4000 $P $rho $type &
python $script_name 3 $dataset_root 4000 $P $rho $type &
python $script_name 4 $dataset_root 4000 $P $rho $type &
python $script_name 5 $dataset_root 4000 $P $rho $type &
python $script_name 6 $dataset_root 4000 $P $rho $type &
python $script_name 7 $dataset_root 4000 $P $rho $type &
python $script_name 8 $dataset_root 4000 $P $rho $type &
python $script_name 9 $dataset_root 4000 $P $rho $type &
python $script_name 10 $dataset_root 4000 $P $rho $type &
python $script_name 11 $dataset_root 4000 $P $rho $type &
python $script_name 12 $dataset_root 4000 $P $rho $type &
python $script_name 13 $dataset_root 4000 $P $rho $type &
python $script_name 14 $dataset_root 4000 $P $rho $type &
python $script_name 15 $dataset_root 4000 $P $rho $type &
python $script_name 16 $dataset_root 4000 $P $rho $type &
python $script_name 17 $dataset_root 4000 $P $rho $type &
python $script_name 18 $dataset_root 4000 $P $rho $type &
python $script_name 19 $dataset_root 4000 $P $rho $type &
python $script_name 20 $dataset_root 4000 $P $rho $type &
python $script_name 21 $dataset_root 4000 $P $rho $type &
python $script_name 22 $dataset_root 4000 $P $rho $type &
python $script_name 23 $dataset_root 4000 $P $rho $type &
python $script_name 24 $dataset_root 4000 $P $rho $type &
python $script_name 25 $dataset_root 4000 $P $rho $type &
python $script_name 26 $dataset_root 4000 $P $rho $type &
python $script_name 27 $dataset_root 4000 $P $rho $type &
python $script_name 28 $dataset_root 4000 $P $rho $type &
python $script_name 29 $dataset_root 4000 $P $rho $type &
wait
