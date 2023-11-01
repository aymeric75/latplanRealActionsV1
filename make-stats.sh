#!/bin/bash

pwdd=$(pwd)




# ##################
# #  BLOCKSWORLD   #
# ##################

# exec 1>myblocks_MAKESTATSTEST.out 2>myblocks_MAKESTATSTEST.err

# task="blocks"
# type="cylinders-4-flat"
# width_height=""
# nb_examples="20000"
# suffix="without" # = without author's weight, no noisy test init/goal
# baselabel="blocks_"$suffix
# after_sample="blocks_cylinders-4-flat_20000_CubeSpaceAE_AMA4Conv_kltune2"
# pb_subdir="prob-cylinders-4"
# conf_folder="06-03T07:41:05.048"
# label="blocks_877"

# domains_dirs=samples/$after_sample/gridAll/logs
# probs_subdir="prob-cylinders-4"



# # ##################
# # #     SOKOBAN    #
# # ##################

# exec 1>mysokoban_MAKESTATS-0-1000inv.out 2>mysokoban_MAKESTATS-0-1000inv.err
# #exec 1>mysokoban_MAKESTATS-194WithEnforce.out 2>mysokoban_MAKESTATS-194WithEnforce.err


# task="sokoban"
# type="sokoban_image-20000-global-global-2-train"
# #type="sokoban_image-20000-global-global-4-train"
# width_height=""
# nb_examples="20000"
# suffix="without"
# baselabel="sokoban"
# after_sample="sokoban_sokoban_image-20000-global-global-2-train_20000_CubeSpaceAE_AMA4Conv_kltune2"
# #after_sample="sokoban_sokoban_image-20000-global-global-4-train_20000_CubeSpaceAE_AMA4Conv_kltune2"
# probs_subdir="sokoban-2-False"



# domains_dirs=samples/$after_sample/logs
# #domains_dirs=samples/$after_sample/gridAll/logs
# #domains_dirs=samples/$after_sample/enforced/logs


# #current_dir="05-12T08:19:26.317"
# # current_dir="05-10T07:11:24.761"
# # current_dir="06-03T07:41:05.048"
# current_dir="05-12T00:12:03.697-0-1000inv"




#################
## PUZZLE MNIST


exec 1>mymnist_052-real.out 2>mymnist_052-real.err

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

conf_folder="05-06T11:21:55.052"
domains_dirs=samples/$after_sample/logs

#conf_folder="05-06T11:21:55.052SEED2"
#conf_folder=05-06T11:21:55.052WITHBOTH
#conf_folder=05-06T11:21:55.052WITHVARS

probs_subdir="puzzle-mnist-3-3"

label=mnist_without_052




# # # 0) displays the stats regarding mutexes and variables found in a sas file
# ./train_kltune.py stats $task $type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2 --hash $current_dir

# exit 0

# # 1) Put all the probs to already filled subdirectories (the ones where you have the domain.pddl)
# for dir in $domains_dirs/*/
# do
#     #echo $(basename $dir)
#     #if [[ "$(basename $dir)" == "05-10T23:39:44.020" ]]; then
#     current_dir=$(basename $dir)
#     echo $dir

#     # echo problem-generators/backup-propositional/vanilla/$probs_subdir
#     # cp -r problem-generators/backup-propositional/vanilla/$probs_subdir $dir
#     # echo $dir/$probs_subdir
#     # cd $dir/$probs_subdir
#     # find . -type f -not -name '*.png' -delete
#     # find . -type f -name 'ama3*' -delete
#     # cd $pwdd

#     #fi
# done

# echo "222"


# # 11) loop and print out the State Variance and the Successor State prediction accuracy
# for dir in $domains_dirs/*/
# do
#     if [[ "$(basename $dir)" == "05-06T11:21:55.052" ]]; then
#         current_dir=$(basename $dir)
#         echo $current_dir

#         #./train_kltune.py states $task $type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2 --hash $current_dir
        
#         ./train_kltune.py stats $task $type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2 --hash $current_dir
        
#     fi
# done

# exit 0




# # 2) Now loop over the configuration folders (in samples/soko.../grid1/logs/), but this time generate the domain.pddl
# for dir in $domains_dirs/*/
# do
#     if [[ "$(basename $dir)" == "05-06T11:21:55.052" ]]; then
#         current_dir=$(basename $dir)
#         echo $current_dir

#         #./train_kltune.py dump $task $type $width_height $nb_examples CubeSpaceAE_AMA4Conv kltune2 --hash $current_dir
        
#         ./pddl-ama3.sh $dir
#     fi
# done

# exit 0



# # 2BIS) generate the problems

# for dir in $domains_dirs/*/
# do

#     if [[ "$(basename $dir)" == "05-06T11:21:55.052" ]]; then
#         echo ${dir}domain.pddl
#         echo $probs_subdir
            
#         echo $dir$probs_subdir

#         for pb in $dir$probs_subdir/*/
#         do
#             ./ama3-planner-problem.py $dir/domain.pddl $pb blind 1
#         done

#     fi
# done
# exit 0







# # CLEAN the PROBLEMS
# for dir in $domains_dirs/*/
# do
#     base_dir=$(basename $dir)
#     if [[ "$base_dir" == "05-11T07:43:01.194WithEnforce" ]]
#     then
#         for pb in $dir$probs_subdir/*/
#         do
#             # echo "pwdd"
#             # echo $pwdd
#             cd $pb
#             res=$(find . -name '*.pddl' | sed 's!.*/!!')
#             echo $res
#             cd $pwdd
#             python $pwdd/downward/src/translate/main.py $dir/domain.pddl $pb/$res --operation='remove_not_from_prob'
#         done
#     fi
# done


# # CLEAN THE DOMAINS
# for dir in $domains_dirs/*/
# do 
#     for pb in $dir$probs_subdir/*/
#     do
#         python $pwdd/downward/src/translate/main.py $dir/domain.pddl $pb --operation='remove_duplicates'
#         exit 0
#     done
# done

#exit 0


# 3) finally again same loop but this time : use the domain.pddl 
#    and go over the problems and try to generate plans for them #   
for dir in $domains_dirs/*/
do
    echo "nouveau"
    #if [[ "$(basename $dir)" == "05-10T23:31:56.595" ]]; then
    echo ${dir}domain.pddl
    echo $probs_subdir
    
    ./ama3-planner-all.py $dir/domain-nopre.pddl $dir/$probs_subdir blind 1
    #fi
done
exit 0


# # 3BIS) generate the plans
# for dir in $domains_dirs/*/
# do

#     domainfile="$dir/domain.pddl"

#     for pb in $dir$probs_subdir/*/
#     do
#         echo $(pwd)
#         echo $pb

#         cd $pb
#         problemfile=$pb$(ls | grep -v "time" | grep problem.pddl)

#         echo $problemfile

#         planfile=$pb/the.plan
#         # echo "icii "
#         # echo $planfile
#         # echo $domainfile 
#         # echo $problemfile
#         ouut=$(/workspace/latplan/latplanClonedAfter/pypy3.8-v7.3.11-linux64/bin/pypy /workspace/latplan/latplanClonedAfter/downward/fast-downward.py --alias lama-first --plan-file "/workspace/latplan/latplanClonedAfter/"$planfile "/workspace/latplan/latplanClonedAfter/"$domainfile "/workspace/latplan/latplanClonedAfter/"$problemfile --translate-options --invariant-generation-max-candidates 0)
#         echo $ouut
#         #/workspace/latplan/latplanClonedAfter/small-launch.sh $planfile $domainfile $problemfile
#         cd ..


#     done
#     ## START PLANNING AND GENERATE (or not) the .PLAN file
#     #
#     #self.bar.update(epoch+1, status = "[finished planning] "+"\n")

# done



########################################################
### Generate The plan file AND THE RECONS OF THE PLANs
########################################################

# loop over config folders (06-*)
for dir in $domains_dirs/*/
do
    # 
    # --plan-file
    base_dir=$(basename $dir)
    #echo $base_dir


    # if [[ "$base_dir" != "05-10T07:04:12.868" ]] && \
    # [[ "$base_dir" != "05-10T07:11:24.761" ]] && \
    # [[ "$base_dir" != "05-10T07:16:09.115" ]] && \
    # [[ "$base_dir" != "05-10T15:21:35.638" ]] && \
    # [[ "$base_dir" != "05-10T15:23:23.735" ]] && \
    # [[ "$base_dir" != "05-10T23:31:56.595" ]] && \
    # [[ "$base_dir" != "05-10T23:39:44.020" ]] && \
    # [[ "$base_dir" != "05-11T07:43:01.194" ]] && \
    # [[ "$base_dir" != "05-11T07:49:57.972" ]] && \
    # [[ "$base_dir" != "05-11T16:00:41.647" ]] && \
    # [[ "$base_dir" != "05-11T16:01:42.528" ]] && \
    # [[ "$base_dir" != "05-12T00:10:43.781" ]] && \
    # [[ "$base_dir" != "05-12T00:12:03.697" ]] && \
    # [[ "$base_dir" != "05-12T08:18:39.774" ]] && \
    # [[ "$base_dir" != "05-12T08:19:26.317" ]] && \
    # [[ "$base_dir" != "05-12T16:30:01.510" ]] && \
    # [[ "$base_dir" != "05-12T16:40:58.399" ]]
    # then
    if [[ "$base_dir" == "05-11T07:43:01.194WithEnforce" ]]
    then
        domain_file=$dir/domain.pddl

        echo "current directory is $dir"
        for pb in $dir$probs_subdir/*/
        do
            problem_file=$pb/*.pddl
            #echo $problem_file
            python downward/fast-downward.py --search-time-limit 3m --plan-file $pb/file.plan --sas-file $pb/sasfile.sas $domain_file $problem_file --search "astar(blind())"
            #python downward/fast-downward.py --alias lama-first --plan-file $pb/file.plan $domain_file $problem_file --translate-options --invariant-generation-max-candidates 0
        done

    fi
done

exit 0

# ########################################################
# ## Generate the trace file
# ########################################################

# # loop over config folders (06-*)
# for dir in $domains_dirs/*/
# do

#     domain_file=$dir/domain.pddl

#     echo $dir

#     for pb in $dir$probs_subdir/*/
#     do
#         problem_file=$pb/problem.pddl
#         planfile=$pb/file.plan
#         arrival $domain_file $problem_file $planfile $pb/tracefile.trace
        
#     done

# done




# ########################################################
# ## Plot the plan
# ########################################################
# for dir in $domains_dirs/*/
# do

#     domain_file=$dir/domain.pddl
#     echo $dir

#     for pb in $dir$probs_subdir/*/
#     do
#         # problem_file=$pb/problem.pddl
#         # planfile=$pb/file.plan
#         # arrival $domain_file $problem_file $planfile $pb/tracefile.trace

#         trace_file=$pb/tracefile.trace
#         csv_file=$pb/csvfile.csv
#         pngfile=$pb/pngfile.png

#         ./plot_plan.py $trace_file $csv_file $pngfile

#     done

# done






# # 5) generate the sols.txt
# for dir in $domains_dirs/*/
# do
#     nbsolSeven=0
#     nbsolSevenOtp=0
#     nbsolFourteen=0
#     nbsolFourteenOpt=0
#     nbsolSevenTotal=0

#     #
#     for pb in $dir$probs_subdir/*/
#     do
#         echo $pb
#         # check if there is a TRACE file (echo it)
#         is_sol=$(find $pb -mindepth 0 -type f -name "tracefile.trace" -exec printf x \; | wc -c)
#         echo $is_sol

#         # 
#         if [ $is_sol -eq "1" ];then

#             nbsolSevenTotal=$((nbsolSevenTotal+1))

#             # fileoutput=$(cat $pb/*.plan)
#             # # check if there is an optimal plan
#             # # if current Pb dir is of type 007-
#             # if [[ "$pb" == *"007-"* ]]; then
#             #     nbsolSeven=$((nbsolSeven+1))
#             #     if [[ "$fileoutput" == *"cost = 7"* ]]; then
#             #         nbsolSevenOtp=$((nbsolSevenOtp+1))
#             #         echo "Optimal solution for 7 steps prob"
#             #     fi
#             # fi

#             # if [[ "$pb" == *"014-"* ]]; then
#             #     nbsolFourteen=$((nbsolFourteen+1))
#             #     if [[ "$fileoutput" == *"cost = 14"* ]]; then
#             #         nbsolFourteenOpt=$((nbsolFourteenOpt+1))
#             #         echo "Optimal solution for 14 steps prob"
#             #     fi
#             # fi


#         fi




#     done

#     # write a file containing line after line
#     # nb
#     echo $nbsolSevenTotal > $dir/sols.txt
#     # echo $nbsolSeven > $dir/sols.txt
#     # echo $nbsolSevenOtp >> $dir/sols.txt
#     # echo $nbsolFourteen >> $dir/sols.txt
#     # echo $nbsolFourteenOpt >> $dir/sols.txt


# done


# ### Generate the invariants.txt and the mutexes.txt (in each conf folder)

# for dir in $domains_dirs/*/
# do
#     nbTotalInvs=0
#     nbTotalMutexes=0

#     base_dir=$(basename $dir)
#     #
#     for pb in $dir$probs_subdir/*/
#     do
#         #echo $pb
        
#         # check if there is a TRACE file (echo it)
#         has_sas=$(find $pb -mindepth 0 -type f -name "sasfile.sas" -exec printf x \; | wc -c)
#         #echo $has_sas

#         # 
#         if [ $has_sas -eq "1" ];then

#             #nbsolSevenTotal=$((nbsolSevenTotal+1))

#             #fileoutput=$(cat $pb/*.sas)

#             nb_vars=$(grep -A1 -P '^end_metric$' $pb/*.sas | sed 's/end_metric//')
#             #nb_vars=$(awk '$0 == "end_metric" {i=1;next};i && i++ <= 2' file)

#             # retrieve value between end_variable and begin_mutex_group
#             #nb_mutexes=$(grep -A1 -P '^end_variable$' $pb/*.sas)
#             nb_mutexes=$(awk 'p == "end_variable" && $0+0 == $0; {p = $0}' $pb/*.sas)
            
#             #echo $nb_mutexes

#             nbTotalInvs=$((nbTotalInvs+$nb_vars))
#             nbTotalMutexes=$((nbTotalMutexes+$nb_mutexes))

#         fi

#     done

#     echo "$nbTotalInvs invariants and $nbTotalMutexes mutexes in $base_dir"

#     echo $nbTotalInvs > $dir/invariants.txt
#     echo $nbTotalMutexes > $dir/mutexes.txt

#     # write a file containing line after line
#     # nb
#     #echo $nbsolSevenTotal > $dir/sols.txt
#     # echo $nbsolSeven > $dir/sols.txt
#     # echo $nbsolSevenOtp >> $dir/sols.txt
#     # echo $nbsolFourteen >> $dir/sols.txt
#     # echo $nbsolFourteenOpt >> $dir/sols.txt

# done

# exit 0

# # create the new.json file (in every pb folder)
# # which takes N, beta_d, beta_z from json + add the content of invatiants.txt (or sols.txt)
# # cp -r problem-generators/backup-propositional/vanilla/$probs_subdir domains_dirs

# loop over the configs
for dir in $domains_dirs/*/
do
    # # 
    # nb_sols=$(find $dir -mindepth 1 -type f -name "*.plan" -exec printf x \; | wc -c)
    # echo $nb_sols > nb_sols.txt
    # # 
    ./create-jsons.py $dir
done



