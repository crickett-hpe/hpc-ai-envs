export FP4=True
export FP8=False
export FP4_RECIPE="nvfp4"
export FP8_PARAM_GATHER=False

# TP overlap not supported yet
export TP_COMM_OVERLAP=False

# FP4 recipe
export NVTE_NVFP4_DISABLE_RHT=0
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0

# FP8 attention recipe
export NVTE_DPA_FP8_RECIPE="Float8CurrentScaling"
