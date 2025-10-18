source ~/.asdf/asdf.sh
export PYTHONPATH=.
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python3-config --prefix)/lib"
# uv run ghlobus/export/tflite_export.py --task=GA_FRAMEWISE --modelpath=checkpoints/ga_experiment_mnv2_e39.ckpt --cnn_name=MobileNet_V2 --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm --sequence_length=110
uv run ghlobus/export/tflite_export.py --task=GA --modelpath=checkpoints/ga_experiment_mnv2_e39.ckpt --cnn_name=MobileNet_V2 --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm --sequence_length=50
uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_ga_model_50.tflite
uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=ghlobus_ga_model_opt_50.tflite
# uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=variable_length_model_max110.tflite --sequence_length=50
