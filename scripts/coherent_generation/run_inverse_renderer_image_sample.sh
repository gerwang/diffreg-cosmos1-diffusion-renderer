#!/usr/bin/env bash
# Optional developer environment file; absent on any other machine.
[[ -f "${HOME}/.pbir_env" ]] && source "${HOME}/.pbir_env"
# Single-sample inverse renderer, image-by-image
case_name=$1
dataset_root="asset/examples/dtc_frames_image"
output_root="asset/example_results/dtc_frames_image"

# Set environment variables.
# CUDA_HOME must point at the env this script is running in — that is
# where nvcc and the CUDA headers the extensions compile against live.
# When launched via `conda run -n <env>`, CONDA_PREFIX already is it;
# hard-coding a path off CONDA_ROOT only works on the machine that set it.
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
if [[ -z "${CUDA_HOME}" ]]; then
  echo "CUDA_HOME is unset and no conda env is active; activate the cosmos env" >&2
  exit 1
fi
export PYTHONPATH="$(pwd)"  # same as ${workspaceFolder}

# Optional: activate conda environment if desired
# source ${CONDA_ROOT}/bin/activate cosmos-predict1.1

if [[ -z "${case_name}" ]]; then
  echo "Usage: $0 <case_name>" >&2
  exit 1
fi

# Run inference script
python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
  --checkpoint_dir checkpoints \
  --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
  --dataset_path "${dataset_root}/${case_name}" \
  --num_video_frames 1 \
  --overlap_n_frames 0 \
  --group_mode image \
  --video_save_folder "${output_root}/" \
  --clip_name_prefix "${case_name}" \
  --skip_existing True \
  --save_video False \
  --chunk_mode all
  #--inference_passes basecolor normal roughness metallic 
