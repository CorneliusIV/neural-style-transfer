import cv2
import os
import time
import errno
import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from stylize import stylize


def build_parser():
    parser = ArgumentParser()

    parser.add_argument(
        '--verbose', action='store_true',
        help='Boolean flag indicating if statements should be printed to the console.')
    parser.add_argument(
        '--style_imgs', nargs='+', type=str, required=True)
    parser.add_argument(
        '--img_name', type=str, default='result',
        help='Filename of the output image.')

    parser.add_argument(
        '--style_imgs_weights', nargs='+', type=float,
        default=[1.0],
        help='Interpolation weights of each of the style images. (example: 0.5 0.5)')

    parser.add_argument(
        '--content_img', type=str,
        help='Filename of the content image (example: lion.jpg)')

    parser.add_argument(
        '--style_imgs_dir', type=str,
        default='./styles',
        help='Directory path to the style images. (default: %(default)s)')

    parser.add_argument(
        '--content_img_dir', type=str,
        default='./image_input',
        help='Directory path to the content image. (default: %(default)s)')

    parser.add_argument(
        '--init_img_type', type=str,
        default='content',
        choices=['random', 'content', 'style'],
        help='Image used to initialize the network. (default: %(default)s)')

    parser.add_argument(
        '--max_size', type=int,
        default=512,
        help='Maximum width or height of the input images. (default: %(default)s)')

    parser.add_argument(
        '--content_weight', type=float,
        default=5e0,
        help='Weight for the content loss function. (default: %(default)s)')

    parser.add_argument(
        '--style_weight', type=float,
        default=1e4,
        help='Weight for the style loss function. (default: %(default)s)')

    parser.add_argument(
        '--tv_weight', type=float,
        default=1e-3,
        help='Weight for the total variational loss function. Set small (e.g. 1e-3). (default: %(default)s)')

    parser.add_argument(
        '--temporal_weight', type=float,
        default=2e2,
        help='Weight for the temporal loss function. (default: %(default)s)')

    parser.add_argument(
        '--content_loss_function', type=int,
        default=1,
        choices=[1, 2, 3],
        help='Different constants for the content layer loss function. (default: %(default)s)')

    parser.add_argument(
        '--content_layers', nargs='+', type=str,
        default=['conv4_2'],
        help='VGG19 layers used for the content image. (default: %(default)s)')

    parser.add_argument(
        '--style_layers', nargs='+', type=str,
        default=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
        help='VGG19 layers used for the style image. (default: %(default)s)')

    parser.add_argument(
        '--content_layer_weights', nargs='+', type=float,
        default=[1.0],
        help='Contributions (weights) of each content layer to loss. (default: %(default)s)')

    parser.add_argument(
        '--style_layer_weights', nargs='+', type=float,
        default=[0.2, 0.2, 0.2, 0.2, 0.2],
        help='Contributions (weights) of each style layer to loss. (default: %(default)s)')

    parser.add_argument(
        '--original_colors', action='store_true',
        help='Transfer the style but not the colors.')

    parser.add_argument(
        '--color_convert_type', type=str,
        default='yuv',
        choices=['yuv', 'ycrcb', 'luv', 'lab'],
        help='Color space for conversion to original colors (default: %(default)s)')

    parser.add_argument(
        '--color_convert_time', type=str,
        default='after',
        choices=['after', 'before'],
        help='Time (before or after) to convert to original colors (default: %(default)s)')

    parser.add_argument(
        '--noise_ratio', type=float,
        default=1.0,
        help="Interpolation value between the content image and noise image if the network is initialized with 'random'.")

    parser.add_argument(
        '--seed', type=int,
        default=0,
        help='Seed for the random number generator. (default: %(default)s)')

    parser.add_argument(
        '--model_weights', type=str,
        default='imagenet-vgg-verydeep-19.mat',
        help='Weights and biases of the VGG-19 network.')

    parser.add_argument(
        '--pooling_type', type=str,
        default='avg',
        choices=['avg', 'max'],
        help='Type of pooling in convolutional neural network. (default: %(default)s)')

    parser.add_argument(
        '--device', type=str,
        default='/gpu:0',
        choices=['/gpu:0', '/cpu:0'],
        help='GPU or CPU mode.  GPU mode requires NVIDIA CUDA. (default|recommended: %(default)s)')

    parser.add_argument(
        '--img_output_dir', type=str,
        default='./image_output',
        help='Relative or absolute directory path to output image and data.')

    # optimizations
    parser.add_argument(
        '--optimizer', type=str,
        default='lbfgs',
        choices=['lbfgs', 'adam'],
        help='Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. (default|recommended: %(default)s)')

    parser.add_argument(
        '--learning_rate', type=float,
        default=1e0,
        help='Learning rate parameter for the Adam optimizer. (default: %(default)s)')

    parser.add_argument(
        '--max_iterations', type=int,
        default=1000,
        help='Max number of iterations for the Adam or L-BFGS optimizer. (default: %(default)s)')

    parser.add_argument(
        '--print_iterations', type=int,
        default=50,
        help='Number of iterations between optimizer print statements. (default: %(default)s)')
    return parser


def maybe_make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def preprocess(img):
    # bgr to rgb
    img = img[..., ::-1]
    # shape (h, w, d) to (1, h, w, d)
    img = img[np.newaxis, :, :, :]
    img -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return img


def get_noise_image(args, content_img):
    np.random.seed(args.seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = args.noise_ratio * noise_img + (1.-args.noise_ratio) * content_img
    return img


def get_init_image(args, content_img, style_imgs, frame=None):
    init_type = args.init_img_type
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(args, content_img)
        return init_img
    return init_img


def get_style_images(args, content_img):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in args.style_imgs:
        path = os.path.join(args.style_imgs_dir, style_fn)
        # bgr image
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(errno.ENOENT, "No such file", path)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs


def get_content_image(args):
    path = os.path.join(args.content_img_dir, args.content_img)
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise OSError(errno.ENOENT, "No such file", path)

    img = img.astype(np.float32)
    h, w, d = img.shape
    mx = args.max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img


def postprocess(img):
    img += np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    # shape (1, h, w, d) to (h, w, d)
    img = img[0]
    img = np.clip(img, 0, 255).astype('uint8')
    # rgb to bgr
    img = img[..., ::-1]
    return img


def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)


def write_image_output(args, output_img, content_img, style_imgs, init_img):
    out_dir = os.path.join(args.img_output_dir, args.img_name)
    maybe_make_directory(out_dir)
    img_path = os.path.join(out_dir, args.img_name+'.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)
    index = 0
    for style_img in style_imgs:
        path = os.path.join(out_dir, 'style_'+str(index)+'.png')
        write_image(path, style_img)
        index += 1
    # save the configuration settings
    out_file = os.path.join(out_dir, 'meta_data.txt')
    f = open(out_file, 'w')
    f.write('image_name: {}\n'.format(args.img_name))
    f.write('content: {}\n'.format(args.content_img))
    index = 0
    for style_img, weight in zip(args.style_imgs, args.style_imgs_weights):
        f.write('styles['+str(index)+']: {} * {}\n'.format(weight, style_img))
        index += 1
    index = 0
    f.write('init_type: {}\n'.format(args.init_img_type))
    f.write('content_weight: {}\n'.format(args.content_weight))
    f.write('style_weight: {}\n'.format(args.style_weight))
    f.write('tv_weight: {}\n'.format(args.tv_weight))
    f.write('content_layers: {}\n'.format(args.content_layers))
    f.write('style_layers: {}\n'.format(args.style_layers))
    f.write('optimizer_type: {}\n'.format(args.optimizer))
    f.write('max_iterations: {}\n'.format(args.max_iterations))
    f.write('max_image_size: {}\n'.format(args.max_size))
    f.close()


def main():
    parser = build_parser()
    args = parser.parse_args()
    content_img = get_content_image(args)
    style_imgs = get_style_images(args, content_img)
    with tf.Graph().as_default():
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        init_img = get_init_image(args, content_img, style_imgs)
        tick = time.time()
        output_image = stylize(args, content_img, style_imgs, init_img)
        write_image_output(args, output_image, content_img, style_imgs, init_img)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))


if __name__ == '__main__':
    main()
