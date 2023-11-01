#!/bin/bash



#################
## PUZZLE MNIST

exec 1>mymnist_052REAL-1.out 2>mymnist_052REAL-1.err


task="puzzle"
type="mnist"
width_height="3 3"
#width_height="4 4"
nb_examples="5000"
#suffix="with" # = with author's weight, no noisy test init/goal
suffix="without" # = without author's weight, no noisy test init/goal
# suffix="noisywith"
# suffix="noisywithout"
baselabel="mnist_"$suffix
after_sample="puzzle_mnist_3_3_5000_CubeSpaceAE_AMA4Conv_kltune2"
pb_subdir="puzzle-mnist-3-3"

conf_folder="05-06T11:21:55.052REAL-1"
#conf_folder="05-06T11:21:55.052SEED2"
#conf_folder=05-06T11:21:55.052WITHBOTH
#conf_folder=05-06T11:21:55.052WITHVARS

label=mnist_without_052
#label=mnist_without_052SEED2
#label=mnist_without_052WITHVARS
#label=mnist_without_052WITHBOTH

##############################################






pwdd=$(pwd)

problem_file="ama3_samples_${after_sample}_logs_${conf_folder}_domain_blind_problem.pddl"
domain_dir=samples/$after_sample/logs/$conf_folder
domain_file=$domain_dir/domain.pddl

problems_dir=problem-generators/backup-propositional/vanilla/$pb_subdir

./train_kltune.py learn $task $type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2 --hash $conf_folder


exit 0