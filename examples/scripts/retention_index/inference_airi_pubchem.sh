#!/bin/bash
# predict_ri_airi.py is provided by the external masskit repository, not this
# repo. It must be installed/on PYTHONPATH in the masskit_ai environment below.
eval "$(mamba shell hook --shell bash)"
mamba activate masskit_ai

INP_FILE="/home/magled/icicle-dev/data/PubChem/PubChem_filtered_for_AIRI.csv"

CKPT_STDNP="/home/magled/icicle-dev/results/airi_models_stdnp/mlruns/676717056023786308/f68e0b0e369e41c9a408645f9d77aaef/artifacts/airi_20260502_012808_f68e0b0e369e41c9a408645f9d77aaef_val_loss=0.0041_epoch=118.ckpt"
CKPT_SEMISTDNP="/home/magled/icicle-dev/results/airi_models_semistdnp/mlruns/985851607513322174/58fe4b07889846e4b90bcf5817fe94f4/artifacts/airi_20260502_012741_58fe4b07889846e4b90bcf5817fe94f4_val_loss=0.0028_epoch=144.ckpt"
CKPT_STDPOLAR="/home/magled/icicle-dev/results/airi_models_stdpolar/mlruns/251871091217423179/f0204cb8b9d647a8ab9c4b03d3f6ecd1/artifacts/airi_20260502_012521_f0204cb8b9d647a8ab9c4b03d3f6ecd1_val_loss=0.0071_epoch=137.ckpt"


python examples/scripts/retention_index/predict_ri_airi.py \
  --input $INP_FILE \
  --output /home/magled/icicle-dev/data/PubChem/AIRI_inference_output_full.tsv \
  --start-row 0 --end-row 94000000 \
  --checkpoint-stdnp $CKPT_STDNP   --checkpoint-semistdnp $CKPT_SEMISTDNP   --checkpoint-stdpolar $CKPT_STDPOLAR  \
  --gpus 0,1,2,3,4,5,6,7 --num-workers 8 --batch-size 512 --keep-temp
