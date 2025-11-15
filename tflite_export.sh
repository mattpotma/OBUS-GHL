source ~/.asdf/asdf.sh
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=""
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python3-config --prefix)/lib"

# export ga
# uv run ghlobus/export/tflite_export.py --task=GA_FRAMEWISE --modelpath=checkpoints/ga_experiment_mnv2_e39.ckpt --cnn_name=MobileNet_V2 --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --sequence_length=110
# uv run ghlobus/export/tflite_export.py --task=GA --modelpath=checkpoints/ga_experiment_mnv2_e39.ckpt --cnn_name=MobileNet_V2 --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --sequence_length=50

# predictions on tflite ga
# uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_ga_model_50.tflite
# uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_ga_model_opt_50.tflite
# uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=variable_length_model_max110.tflite --sequence_length=50

# predictions on gates ga
# uv run ghlobus/inference/inference.py --task=GA --modelpath=checkpoints/ga_experiment_mnv2_e39.ckpt --cnn_name=MobileNet_V2 --file=/home/ella/src/OBUS-GHL/dicoms/test.dcm

# export fp
# uv run ghlobus/export/tflite_export.py --task=FP --modelpath=checkpoints/fp_experiment_e14.ckpt --cnn_name=MobileNet_V2 --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --sequence_length=50

# predictions on tflite fp
# uv run ghlobus/inference/tflite_inference.py --task=FP --tflite_dir=./tflite_models --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_fp_model_50.tflite
# uv run ghlobus/inference/tflite_inference.py --task=FP --tflite_dir=./tflite_models --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_fp_model_opt_50.tflite

# predictions on gates fp
# uv run ghlobus/inference/inference.py --task=FP --modelpath=checkpoints/fp_experiment_e14.ckpt --cnn_name=MobileNet_V2 --file=/home/ella/src/OBUS-GHL/dicoms/test.dcm

# export efw
# uv run ghlobus/export/tflite_export.py --task=EFW --modelpath=checkpoints/efw_experiment_e56.ckpt --cnn_name=MobileNet_V2 --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --sequence_length=50

# predictions on tflite efw
# uv run ghlobus/inference/tflite_inference.py --task=EFW --tflite_dir=./tflite_models --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_efw_model_50.tflite
# uv run ghlobus/inference/tflite_inference.py --task=EFW --tflite_dir=./tflite_models --dicom=/home/ella/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_efw_model_opt_50.tflite

#predictions on gates efw
# uv run ghlobus/inference/inference.py --task=EFW --modelpath=checkpoints/efw_experiment_e56.ckpt --cnn_name=MobileNet_V2 --file=/home/ella/src/OBUS-GHL/dicoms/test.dcm
