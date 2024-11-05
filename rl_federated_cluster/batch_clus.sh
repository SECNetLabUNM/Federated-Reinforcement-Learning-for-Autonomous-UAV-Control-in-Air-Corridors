#!/bin/bash
# begin train for hexgon release



source /home/meng/miniconda3/bin/activate torch2

# Current time for logging or file naming
current_time=$(date +"%Y%m%d%H%M%S")
desired_directory="/home/meng/Documents/Code/HTransRL/expfl2"

# Create directory if it doesn't exist
if [ ! -d "$desired_directory" ]; then
    mkdir -p "$desired_directory"
fi

cd "$desired_directory"

# Parameters initialization
param1_values=(9) # num of agents
param2_values=(1) # Mult horizon
param3_values=(1) # Mult batch
param4_values=('True')  # token query, work for trans only
param5_values=('True' )  # reduce state or original state, original state is not well maintained
#param6_values=(3 6)
param6_values=(6.0) # visibility
param8_values=(19) #
param9_values=('fc10_3e') # network model
param10_values=(True)  # whether to share layers
param11_values=(1.5e-5)  # Alr
param12_values=(1.5e-6)  # Clr
param13_values=(0.3)  # Turb var
param14_values=(0.3 )  # acceleration max
param15_values=( 128 )  # Num neurons
param16_values=( 'False')  # Partial fine tune
param17_values=(1)  # Fed every
param18_values=( 'all' )  # Fedkey
param19_values=(4)  # K epoch
max_concurrent=100
concurrent_processes=0
num_executions=1

# Task distribution counters
num_task_gpu0=0
num_task_gpu1=0

# Main loop for parameter combination execution
for i in $(seq $num_executions); do
  for param1 in "${param1_values[@]}"; do
    for param2 in "${param2_values[@]}"; do
      for param3 in "${param3_values[@]}"; do
        for param4 in "${param4_values[@]}"; do
          for param5 in "${param5_values[@]}"; do
            for param6 in "${param6_values[@]}"; do
                for param8 in "${param8_values[@]}"; do
                  for param9 in "${param9_values[@]}"; do
                    for param10 in "${param10_values[@]}"; do
                      for param11 in "${param11_values[@]}"; do
                        for param12 in "${param12_values[@]}"; do
                          for param13 in "${param13_values[@]}"; do
                            for param14 in "${param14_values[@]}"; do
                              for param15 in "${param15_values[@]}"; do
                                for param16 in "${param16_values[@]}"; do
                                  for param17 in "${param17_values[@]}"; do
                                    for param18 in "${param18_values[@]}"; do
                                      for param19 in "${param19_values[@]}"; do
#                        if awk -v p11="$param11" -v p12="$param12" 'BEGIN{ exit !(p11 == 1e-5 && p12 == 1.1) }'; then
#                          continue  # Skip this combination
#                        fi

                      gpu_index=0 # Set default GPU indexn

                      # Construct experiment name
                      exp_name="new_net:${param1}agents_${param6}Visibility_${param13}Turb_${param17}Fed_evry"
                      echo $PATH
                      # Run the Python script with parameters
                      CUDA_VISIBLE_DEVICES=$gpu_index python /home/meng/Documents/Code/FL-HtransL/rl_federated_cluster/main_cluster.py \
                          --seed 33 \
                          --exp-name ${exp_name} \
                          --num_agents ${param1} \
                          --multiply_horrizion ${param2} \
                          --multiply_batch ${param3} \
                          --visibility ${param6} \
                          --level ${param8} \
                          --net_model ${param9} \
                          --a_lr ${param11} \
                          --c_lr ${param12} \
                          --turbulence_variance ${param13} \
                          --partial_fine_tune ${param16} \
                          --fed_every ${param17} \
                          --fed_key  ${param18} \
                          --K_epochs ${param19} \
                          --Max_train_steps 1e6 \
                          --LoadModel True \
 #--current_time "$current_time" \
                      # Manage concurrent processes
                      concurrent_processes=$((concurrent_processes + 1))

                      # Limit the number of concurrent processes
                      if [ "$concurrent_processes" -ge "$max_concurrent" ]; then
                        wait
                        concurrent_processes=0
                      fi
                      done
                      done
                      done
                      done
                      done
                      done
                      done
                      done

                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# Wait for all background processes to finish
wait