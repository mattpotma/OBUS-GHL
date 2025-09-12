export PYTHONPATH=.
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(python3-config --prefix)/lib"
uv run ghlobus/export/tflite_export.py --task=GA --modelpath=checkpoints/ga_experiment_mnv2_e39.ckpt --cnn_name=MobileNet_V2 --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm
uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=unified_model.tflite
uv run ghlobus/inference/tflite_inference.py --task=GA --tflite_dir=./tflite_models --dicom=/home/matt/src/OBUS-GHL/dicoms/test.dcm --outdir=./test_tflite --tflite_model_name=unified_model_optimized_default.tflite
