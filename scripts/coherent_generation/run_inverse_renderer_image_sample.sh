#!/usr/bin/env bash
source ~/.pbir_env
# Single-sample inverse renderer, image-by-image
case_name=$1
dataset_root="asset/examples/dtc_frames_image"
output_root="asset/example_results/dtc_frames_image"

# Set environment variables
export CUDA_HOME="${CONDA_ROOT}/envs/cosmos-predict1.1"
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
