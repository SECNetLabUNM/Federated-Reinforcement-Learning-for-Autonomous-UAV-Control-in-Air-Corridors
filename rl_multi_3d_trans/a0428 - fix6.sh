#!/bin/bash
# begin train for hexgon release



source /home/kun/anaconda3/bin/activate torch

# Current time for logging or file naming
current_time=$(date +"%Y%m%d%H%M%S")
desired_directory="/mnt/storage/result"

# Create directory if it doesn't exist
if [ ! -d "$desired_directory" ]; then
    mkdir -p "$desired_directory"
fi

cd "$desired_directory"

# Parameters initialization
param1_values=(4) # num of agents
param2_values=(8)
param3_values=(16)
param4_values=('True')  # token query, work for trans only
param5_values=('True' )  # reduce state or original state, original state is not well maintained
#param6_values=(3 6)
param6_values=(2) # num of encoder
param7_values=(2) # num of decoder
param8_values=( 19) #reset level;     level 2: random 1 piece;   level 3: random 2 pieces; level 13: random 3 pieces;
param9_values=('fc10_3e' 'dec') # network model
param10_values=(True)  # whether to share layers
param11_values=(1.0 )  # beta base
param12_values=(1.1)  # beta range
param13_values=(2)  # num of future corridors in state, at least 1
param14_values=(0.3 )  # acceleration max
param15_values=( 128 )  # capability of corridor indexing
param16_values=( 'True' )  # rotate cylinder for simple
param17_values=(2 )  # state choice
param18_values=( 'True' 'False' )  # state choice
param19_values=(4)  # state choice
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
              for param7 in "${param7_values[@]}"; do
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
                        if awk -v p11="$param11" -v p12="$param12" 'BEGIN{ exit !(p11 == 1e-5 && p12 == 1.1) }'; then
                          continue  # Skip this combination
                        fi

                      gpu_index=0 # Set default GPU indexn

                      # Construct experiment name
                      exp_name="new_net:width_${param15}epoch${param19}_corindex${param18}_net${param9}_horizon${param2}_level${param8}_capacity${param1}_beta_adaptor${param12}"
                      echo $PATH
                      # Run the Python script with parameters
                      CUDA_VISIBLE_DEVICES=$gpu_index python /home/kun/PycharmProjects/air-corridor_ncfo/rl_multi_3d_trans/main.py \
                          --seed 55 \
                          --LoadModel False \
                          --variable_agent False\
                          --time ${current_time} \
                          --multiply_horrizion ${param2} \
                          --multiply_batch ${param3} \
                          --token_query ${param4} \
                          --reduce_space ${param5} \
                          --num_enc ${param6} \
                          --num_dec ${param7} \
                          --level ${param8} \
                          --curriculum True  \
                          --net_width ${param15} \
                          --base_difficulty 0.1\
                          --num_agents ${param1} \
                          --beta_base ${param11} \
                          --beta_adaptor_coefficient ${param12} \
                          --net_model ${param9} \
                          --rotate_for_cylinder ${param16} \
                          --state_choice ${param17} \
                          --with_corridor_index  ${param18} \
                          --K_epochs ${param19} \
                          --num_obstacles 4 \
                          --num_ncfos 3 \
                          --Max_train_steps 3e7 \
                          --exp-name ${exp_name}  &

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
done

# Wait for all background processes to finish
wait