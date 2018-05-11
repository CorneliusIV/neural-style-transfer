# neural-style-transfer


This is a combination of https://github.com/cysmith/neural-style-tf and https://github.com/anishathalye/neural-style . I have been using both, but I am now combining the features I like about each one into a single repo as well as some small cleanup.
See their original repos for the full features.

This is a TensorFlow implementation of several techniques described in the papers:
* [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
* [Artistic style transfer for videos](https://arxiv.org/abs/1604.08610)
by Manuel Ruder, Alexey Dosovitskiy, Thomas Brox
* [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/abs/1606.05897)
by Leon A. Gatys, Matthias Bethge, Aaron Hertzmann, Eli Shechtman

Additionally, techniques are presented for semantic segmentation and multiple style transfer.

The Neural Style algorithm synthesizes a [pastiche](https://en.wikipedia.org/wiki/Pastiche) by separating and combining the content of one image with the style of another image using convolutional neural networks (CNN). Below is an example of transferring the artistic style of [The Starry Night](https://en.wikipedia.org/wiki/The_Starry_Night) onto a photograph of an African lion:


## Setup
#### Dependencies:
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [opencv](http://opencv.org/downloads.html)

#### Optional (but recommended) dependencies:
* [CUDA](https://developer.nvidia.com/cuda-downloads) 7.5+
* [cuDNN](https://developer.nvidia.com/cudnn) 5.0+

#### After installing the dependencies:
* Download the [VGG-19 model weights](http://www.vlfeat.org/matconvnet/pretrained/) (see the "VGG-VD models from the *Very Deep Convolutional Networks for Large-Scale Visual Recognition* project" section). More info about the VGG-19 network can be found [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).
* After downloading, copy the weights file `imagenet-vgg-verydeep-19.mat` to the project directory.

## Usage
### Basic Usage

#### Single Image
1. Copy 1 content image to the default image content directory `./image_input`
2. Copy 1 or more style images to the default style directory `./styles`
3. Run the command:
```
bash stylize_image.sh <path_to_content_image> <path_to_style_image>
```
*Example*:
```
bash stylize_image.sh ./image_input/lion.jpg ./styles/kandinsky.jpg
```
*Note*: Supported image formats include: `.png`, `.jpg`, `.ppm`, `.pgm`

*Note*: Paths to images should not contain the `~` character to represent your home directory; you should instead use a relative path or the absolute path.


### Advanced Usage
#### Single Image
1. Copy content images to the default image content directory `./image_input`
2. Copy 1 or more style images to the default style directory `./styles`
3. Run the command with specific arguments:
```
python neural_style.py <arguments>
```
*Example (Single Image)*:
```
python neural_style.py --content_img golden_gate.jpg --style_imgs starry-night.jpg --max_size 1000  --max_iterations 100 --original_colors --device /cpu:0 --verbose;
```

To use multiple style images, pass a *space-separated* list of the image names and image weights like this:

`--style_imgs starry_night.jpg the_scream.jpg --style_imgs_weights 0.5 0.5`

*Note*:  When using `--init_frame_type prev_warp` you must have previously computed the backward and forward optical flow between the frames.  See `./video_input/make-opt-flow.sh` and `./video_input/run-deepflow.sh`

#### Arguments
* `--content_img`: Filename of the content image. *Example*: `lion.jpg`
* `--content_img_dir`: Relative or absolute directory path to the content image. *Default*: `./image_input`
* `--style_imgs`: Filenames of the style images. To use multiple style images, pass a *space-separated* list.  *Example*: `--style_imgs starry-night.jpg`
* `--style_imgs_weights`: The blending weights for each style image.  *Default*: `1.0` (assumes only 1 style image)
* `--style_imgs_dir`: Relative or absolute directory path to the style images. *Default*: `./styles`
* `--init_img_type`: Image used to initialize the network. *Choices*: `content`, `random`, `style`. *Default*: `content`
* `--max_size`: Maximum width or height of the input images. *Default*: `512`
* `--content_weight`: Weight for the content loss function. *Default*: `5e0`
* `--style_weight`: Weight for the style loss function. *Default*: `1e4`
* `--tv_weight`: Weight for the total variational loss function. *Default*: `1e-3`
* `--temporal_weight`: Weight for the temporal loss function. *Default*: `2e2`
* `--content_layers`: *Space-separated* VGG-19 layer names used for the content image. *Default*: `conv4_2`
* `--style_layers`: *Space-separated* VGG-19 layer names used for the style image. *Default*: `relu1_1 relu2_1 relu3_1 relu4_1 relu5_1`
* `--content_layer_weights`: *Space-separated* weights of each content layer to the content loss. *Default*: `1.0`
* `--style_layer_weights`: *Space-separated* weights of each style layer to loss. *Default*: `0.2 0.2 0.2 0.2 0.2`
* `--original_colors`: Boolean flag indicating if the style is transferred but not the colors.
* `--color_convert_type`: Color spaces (YUV, YCrCb, CIE L\*u\*v\*, CIE L\*a\*b\*) for luminance-matching conversion to original colors. *Choices*: `yuv`, `ycrcb`, `luv`, `lab`. *Default*: `yuv`
* `--style_mask`: Boolean flag indicating if style is transferred to masked regions.
* `--style_mask_imgs`: Filenames of the style mask images (example: `face_mask.png`). To use multiple style mask images, pass a *space-separated* list.  *Example*: `--style_mask_imgs face_mask.png face_mask_inv.png`
* `--noise_ratio`: Interpolation value between the content image and noise image if network is initialized with `random`. *Default*: `1.0`
* `--seed`: Seed for the random number generator. *Default*: `0`
* `--model_weights`: Weights and biases of the VGG-19 network.  Download [here](http://www.vlfeat.org/matconvnet/pretrained/). *Default*:`imagenet-vgg-verydeep-19.mat`
* `--pooling_type`: Type of pooling in convolutional neural network. *Choices*: `avg`, `max`. *Default*: `avg`
* `--device`: GPU or CPU device.  GPU mode highly recommended but requires NVIDIA CUDA. *Choices*: `/gpu:0` `/cpu:0`. *Default*: `/gpu:0`
* `--img_output_dir`: Directory to write output to.  *Default*: `./image_output`
* `--img_name`: Filename of the output image. *Default*: `result`
* `--verbose`: Boolean flag indicating if statements should be printed to the console.

#### Optimization Arguments
* `--optimizer`: Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. *Choices*: `lbfgs`, `adam`. *Default*: `lbfgs`
* `--learning_rate`: Learning-rate parameter for the Adam optimizer. *Default*: `1e0`

<p align="center">
<img src="examples/equations/plot.png" width="360px">
</p>

* `--max_iterations`: Max number of iterations for the Adam or L-BFGS optimizer. *Default*: `1000`
* `--print_iterations`: Number of iterations between optimizer print statements. *Default*: `50`
* `--content_loss_function`: Different constants K in the content loss function. *Choices*: `1`, `2`, `3`. *Default*: `1`

## Questions and Errata

Send questions or issues:
<img src="examples/equations/email.png">

## Memory
By default, `neural-style-tf` uses the NVIDIA cuDNN GPU backend for convolutions and L-BFGS for optimization.
These produce better and faster results, but can consume a lot of memory. You can reduce memory usage with the following:

* **Use Adam**: Add the flag `--optimizer adam` to use Adam instead of L-BFGS. This should significantly
  reduce memory usage, but will require tuning of other parameters for good results; in particular you should
  experiment with different values of `--learning_rate`, `--content_weight`, `--style_weight`
* **Reduce image size**: You can reduce the size of the generated image with the `--max_size` argument.

## Implementation Details
All images were rendered on a machine with:
* **CPU:** Intel Core i7-6800K @ 3.40GHz × 12
* **GPU:** NVIDIA GeForce GTX 1080/PCIe/SSE2
* **OS:** Linux Ubuntu 16.04.1 LTS 64-bit
* **CUDA:** 8.0
* **python:** 2.7.12
* **tensorflow:** 0.10.0rc
* **opencv:** 2.4.9.1

## Acknowledgements

The implementation is based on the projects:
* Torch (Lua) implementation 'neural-style' by [jcjohnson](https://github.com/jcjohnson)
* Torch (Lua) implementation 'artistic-videos' by [manuelruder](https://github.com/manuelruder)


## Citation

```
@misc{Smith2016,
  author = {Smith, Cameron},
  title = {neural-style-tf},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cysmith/neural-style-tf}},
}

@misc{athalye2015neuralstyle,
  author = {Anish Athalye},
  title = {Neural Style},
  year = {2015},
  howpublished = {\url{https://github.com/anishathalye/neural-style}},
  note = {commit xxxxxxx}
}
```
