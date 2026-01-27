source ~/.asdf/asdf.sh
export PYTHONPATH=.
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python3-config --prefix)/lib"
export DICOM=/home/matt/src/OBUS-GHL/dicoms/test.dcm
export SEQ=50

# GA Export, inference, tflite inference
export GA_CKPT=checkpoints/ga_experiment_mnv2_e39.ckpt
uv run ghlobus/export/tflite_export.py --task=GA --modelpath=$GA_CKPT --cnn_name=MobileNet_V2 --dicom=$DICOM --sequence_length=$SEQ
uv run ghlobus/inference/inference.py --task=GA --modelpath=$GA_CKPT --cnn_name=MobileNet_V2 --file=$DICOM --sequence_length=$SEQ
uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=$DICOM --outdir=./test_tflite --tflite_model_name=ghlobus_ga_model_50.tflite --sequence_length=$SEQ
uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=$DICOM --outdir=./test_tflite --tflite_model_name=ghlobus_ga_model_opt_50.tflite --sequence_length=$SEQ

# FP Export, inference, tflite inference
export FP_CKPT=checkpoints/fp_experiment_e14.ckpt
uv run ghlobus/export/tflite_export.py --task=FP --modelpath=$FP_CKPT --cnn_name=MobileNet_V2 --dicom=$DICOM --sequence_length=$SEQ
uv run ghlobus/inference/inference.py --task=FP --modelpath=$FP_CKPT --cnn_name=MobileNet_V2 --file=$DICOM --sequence_length=$SEQ
uv run ghlobus/inference/tflite_inference.py --task=FP --tflite_dir=./tflite_models --dicom=$DICOM --outdir=./test_tflite --tflite_model_name=ghlobus_fp_model_50.tflite --sequence_length=$SEQ
uv run ghlobus/inference/tflite_inference.py --task=FP --tflite_dir=./tflite_models --dicom=$DICOM --outdir=./test_tflite --tflite_model_name=ghlobus_fp_model_opt_50.tflite --sequence_length=$SEQ

# EFW Export, inference, tflite inference
export EFW_CKPT=checkpoints/efw_experiment_e56.ckpt
uv run ghlobus/export/tflite_export.py --task=EFW --modelpath=$EFW_CKPT --cnn_name=MobileNet_V2 --dicom=$DICOM --sequence_length=$SEQ
uv run ghlobus/inference/inference.py --task=EFW --modelpath=$EFW_CKPT --cnn_name=MobileNet_V2 --file=$DICOM --sequence_length=$SEQ
uv run ghlobus/inference/tflite_inference.py --task=EFW --tflite_dir=./tflite_models --dicom=$DICOM --outdir=./test_tflite --tflite_model_name=ghlobus_efw_model_50.tflite --sequence_length=$SEQ
uv run ghlobus/inference/tflite_inference.py --task=EFW --tflite_dir=./tflite_models --dicom=$DICOM --outdir=./test_tflite --tflite_model_name=ghlobus_efw_model_opt_50.tflite --sequence_length=$SEQ
