# check whether gurobi is under the path /opt/gurobi/gurobi.lic
# if not alert
if [ ! -f /opt/gurobi/gurobi.lic ]; then
    echo "Gurobi license file not found under /opt/gurobi/gurobi.lic"
    echo "Please make sure that the gurobi license file is under /opt/gurobi/gurobi.lic"
    exit 1
fi

###############################
#
# Main experiment case 1-10
#
###############################
# case 1
model_size=125m
device_names=("Tesla_V100-SXM2-32GB") 
device_numbers=(1)  # define device numbers as a list of integers
OMEGA_FILE=${ROOT_DIR}/scripts/accuracyPPL/generated_ind/gen_opt_125m_ind.pkl

llm_pq-algo --model_size ${model_size} \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --omega_file $OMEGA_FILE --ilp_seed 120 \
 --theta 1 --global_bz 32 --fit 