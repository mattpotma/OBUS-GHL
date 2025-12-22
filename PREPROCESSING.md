# DICOM to TFLite Model preprocessing for Gestational Age model

This is my attempt to explain the way that the GHL OBUS preprocesses data to input to the TFLite model. Most of the
weird stuff in here I blame on them, I tried to make the TFLite model contain as much pre/post processing as I can, but
everything left in here is much easier done outside of the model.

Note that I haven't traced through the DICOM loading in this repo to see what format our images are in. Hopefully since
you're skipping DICOM exporting, a lot of this can be simplified and the details don't matter too much.

## Overview

The preprocessing pipeline takes a DICOM file and produces a `uint8` tensor of shape `(1, N, 3, 256, 256)`, where:

- `1` is the batch dimension
- `N` is the number of frames, hard coded to `50`
- `3` for RGB channels because RGB is cool
- `256x256` for the post-cropping frame dimension

## Model Input/Output details

Below are the input/output details that you should be able to grab from the TFLite from C++. The `quantization` stuff
may differ, but I don't think that's relevant. There's only the one input and one output.

```
Input details: [{
    'name': 'serving_default_args_0:0',
    'index': 0,
    'shape': array([  1,  50,   3, 256, 256], dtype=int32),
    'shape_signature': array([  1,  50,   3, 256, 256], dtype=int32),
    'dtype': <class 'numpy.uint8'>,
    'quantization': (0.0, 0),
    'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0},
    'sparsity_parameters': {}
}]

Output details: [{
    'name': 'StatefulPartitionedCall:0',
    'index': 226,
    'shape': array([], dtype=int32),
    'shape_signature': array([], dtype=int32),
    'dtype': <class 'numpy.float32'>,
    'quantization': (0.0, 0),
    'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32),
    'quantized_dimension': 0},
    'sparsity_parameters': {}
}]
```

## Pipeline steps

### Load DICOM file

**Location**: `ghlobus/utilities/inference_utils.py::get_dicom_frames`

This calls a `ghlobus/utilities/data_utils.py::preprocess_video` which has lots of annoying branching in it that I
haven't traced through.

I believe at this point the frames are converted to RGB space. In our case, simply tile the grayscale frame 3x. If our
post-scan is 500x500, this becomes `3x500x500`. Note that in our other models, we have channels as the last dim (i.e.
`500x500x3`), this repo (and PyTorch) work with "Channels first", or `3x500x500`. This doesn't really matter until the
input to the TFLite model, which expects the dimensions above.

### Subsampling frames in time

For inference, we only want 50 frames. If there are fewer than 50 frames in a sweep, we should tell the user to retake
it. If more than 50, we want to periodically sample frames out of that. I do it like this in Python:

```python
if sequence_length is not None and frames_torch.shape[1] > sequence_length:
    frames_torch = frames_torch[:, :sequence_length, :, :, :]
    # Subsample frames evenly across the temporal dimension
    indices = np.linspace(0, frames_torch.shape[1] - 1, sequence_length).astype(int)
    frames_torch = frames_torch[:, indices, :, :, :]
```

And that translates to something like this in the Lord's language:

```c
if (num_frames > sequence_length) {
    for (i = 0; i < sequence_length; i++) {
        index = (i * (num_frames - 1)) / (sequence_length - 1);
        new_frames[i] = frames[index];
    }
    num_frames = sequence_length;
}
```

You should now have 50 frames to run through the model.

### Crop and Scale Frames

**Location**: `ghlobus/utilities/data_utils.py::crop_and_scale_frames_tensor`

**Crop to US Region**

It seems like the DICOM file contains a region where the ultrasound image is valid, called `SequenceOfUltasoundRegions`
which is unpacked in `ghlobus/utilities/data_utils.py::get_crop_box`. The image is cropped to these bounds.

**Scale to alpha**

The image is now scaled to a target ratio of `0.75 mm/pixel` further into the `crop_and_scale_frames_tensor` function.
Bilinear interpolation and antialiasing, although the latter I don't think matters. Just stick to any bilinear interp.

**Pad to minimum dimensions**

If, after the physical scaling occurs, either dimension of the image is less than 288px, zero pad it to 288. The image
should now be a minimum of 288x288px.

### Center crop to 256x256 and add fake news batch dimension

Now take the center crop of the image to `256x256`. This should give a 19.2x19.2cm window.

Finally, I don't know if adding a dimension of 1 is a real thing in C++, but python needs it sometimes. If you have
already stacked all of the temporal frames, then the dimensions of the tensor should be `1, 50, 3, 256, 256`, with a
data type of `uint8`.

### TFLite Inference

**Location**: `ghlobus/inference/tflite_inference.py::tflite_inference`

Input is the `uint8` `[1,50,3,256,256]` tensor. Only one input. Output is gestational days. Only one output.

I think the C++ bindings look a lot like this.

```python
interpreter.set_tensor(input_details[0]["index"], input_quantized)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])
```

### Handling the 6 sweeps

For the imaging protocol, the user takes 6 sweeps. 3 side to side, 3 top to bottom.

Each of the 6 sweeps should go through the above process, and each of them return a gestational age in days. Run them
all through individually, and average the GA prediction.

We'll probably implement some kind of check that they're all within a reasonable amount (like maybe +/- 10 days?) and
say that the sweep failed if outside of that range. But I think Ella will need to do some testing to see what's
reasonable.

## The two models

I'm giving you two models via teams, `ghlobus_ga_model_50.tflite` and `ghlobus_ga_model_opt_50.tflite`. Either one
should work. The one with `opt` is smaller by about a factor of 4. Slightly less accurate, but will be faster on a lot
of devices. We (Ella) need to do testing to see which is best. But even the big one is only 14 MB and pretty fast.

# Fetal Presentation model

The fetal presentation model has the exact same preprocessing as the gestational age model. It outputs a probability, where anything from 0.5 to 1 is cephalic, and 0 to 0.5 is non-cephalic. This binary result (cephalic vs non-cephalic) is the sole output of the model.