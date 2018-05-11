import numpy as np
import scipy.io
import tensorflow as tf


def conv_layer(layer_name, layer_input, W):
    conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
    return conv


def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    return relu


def pool_layer(args, layer_name, layer_input):
    if args.pooling_type == 'avg':
        pool = tf.nn.avg_pool(
            layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    elif args.pooling_type == 'max':
        pool = tf.nn.max_pool(
            layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return pool


def get_weights(vgg_layers, i):
    weights = vgg_layers[i][0][0][2][0][0]
    W = tf.constant(weights)
    return W


def get_bias(vgg_layers, i):
    bias = vgg_layers[i][0][0][2][0][1]
    b = tf.constant(np.reshape(bias, (bias.size)))
    return b


def build_model(args, input_img):
    print('\nBUILDING VGG-19 NETWORK')
    net = {}
    _, h, w, d = input_img.shape

    vgg_rawnet = scipy.io.loadmat(args.model_weights)
    vgg_layers = vgg_rawnet['layers'][0]
    net['input'] = tf.Variable(np.zeros((1, h, w, d), dtype=np.float32))

    net['conv1_1'] = conv_layer('conv1_1', net['input'], W=get_weights(vgg_layers, 0))
    net['relu1_1'] = relu_layer('relu1_1', net['conv1_1'], b=get_bias(vgg_layers, 0))

    net['conv1_2'] = conv_layer('conv1_2', net['relu1_1'], W=get_weights(vgg_layers, 2))
    net['relu1_2'] = relu_layer('relu1_2', net['conv1_2'], b=get_bias(vgg_layers, 2))

    net['pool1'] = pool_layer(args, 'pool1', net['relu1_2'])

    net['conv2_1'] = conv_layer('conv2_1', net['pool1'], W=get_weights(vgg_layers, 5))
    net['relu2_1'] = relu_layer('relu2_1', net['conv2_1'], b=get_bias(vgg_layers, 5))

    net['conv2_2'] = conv_layer('conv2_2', net['relu2_1'], W=get_weights(vgg_layers, 7))
    net['relu2_2'] = relu_layer('relu2_2', net['conv2_2'], b=get_bias(vgg_layers, 7))

    net['pool2'] = pool_layer(args, 'pool2', net['relu2_2'])

    net['conv3_1'] = conv_layer('conv3_1', net['pool2'], W=get_weights(vgg_layers, 10))
    net['relu3_1'] = relu_layer('relu3_1', net['conv3_1'], b=get_bias(vgg_layers, 10))

    net['conv3_2'] = conv_layer('conv3_2', net['relu3_1'], W=get_weights(vgg_layers, 12))
    net['relu3_2'] = relu_layer('relu3_2', net['conv3_2'], b=get_bias(vgg_layers, 12))

    net['conv3_3'] = conv_layer('conv3_3', net['relu3_2'], W=get_weights(vgg_layers, 14))
    net['relu3_3'] = relu_layer('relu3_3', net['conv3_3'], b=get_bias(vgg_layers, 14))

    net['conv3_4'] = conv_layer('conv3_4', net['relu3_3'], W=get_weights(vgg_layers, 16))
    net['relu3_4'] = relu_layer('relu3_4', net['conv3_4'], b=get_bias(vgg_layers, 16))

    net['pool3'] = pool_layer(args, 'pool3', net['relu3_4'])

    net['conv4_1'] = conv_layer('conv4_1', net['pool3'], W=get_weights(vgg_layers, 19))
    net['relu4_1'] = relu_layer('relu4_1', net['conv4_1'], b=get_bias(vgg_layers, 19))

    net['conv4_2'] = conv_layer('conv4_2', net['relu4_1'], W=get_weights(vgg_layers, 21))
    net['relu4_2'] = relu_layer('relu4_2', net['conv4_2'], b=get_bias(vgg_layers, 21))

    net['conv4_3'] = conv_layer('conv4_3', net['relu4_2'], W=get_weights(vgg_layers, 23))
    net['relu4_3'] = relu_layer('relu4_3', net['conv4_3'], b=get_bias(vgg_layers, 23))

    net['conv4_4'] = conv_layer('conv4_4', net['relu4_3'], W=get_weights(vgg_layers, 25))
    net['relu4_4'] = relu_layer('relu4_4', net['conv4_4'], b=get_bias(vgg_layers, 25))

    net['pool4'] = pool_layer(args, 'pool4', net['relu4_4'])

    net['conv5_1'] = conv_layer('conv5_1', net['pool4'], W=get_weights(vgg_layers, 28))
    net['relu5_1'] = relu_layer('relu5_1', net['conv5_1'], b=get_bias(vgg_layers, 28))

    net['conv5_2'] = conv_layer('conv5_2', net['relu5_1'], W=get_weights(vgg_layers, 30))
    net['relu5_2'] = relu_layer('relu5_2', net['conv5_2'], b=get_bias(vgg_layers, 30))

    net['conv5_3'] = conv_layer('conv5_3', net['relu5_2'], W=get_weights(vgg_layers, 32))
    net['relu5_3'] = relu_layer('relu5_3', net['conv5_3'], b=get_bias(vgg_layers, 32))

    net['conv5_4'] = conv_layer('conv5_4', net['relu5_3'], W=get_weights(vgg_layers, 34))
    net['relu5_4'] = relu_layer('relu5_4', net['conv5_4'], b=get_bias(vgg_layers, 34))

    net['pool5'] = pool_layer(args, 'pool5', net['relu5_4'])

    return net


def get_optimizer(args, loss):
    print_iterations = args.print_iterations if args.verbose else 0
    if args.optimizer == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B',
            options={'maxiter': args.max_iterations,
                     'disp': print_iterations})
    elif args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
    return optimizer


def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


def content_layer_loss(args, p, x):
    _, h, w, d = p.get_shape()
    M = h.value * w.value
    N = d.value
    if args.content_loss_function == 1:
        K = 1. / (2. * N**0.5 * M**0.5)
    elif args.content_loss_function == 2:
        K = 1. / (N * M)
    elif args.content_loss_function == 3:
        K = 1. / 2.
    loss = K * tf.reduce_sum(tf.pow((x - p), 2))
    return loss


def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G


def sum_style_losses(args, sess, net, style_imgs):
    total_style_loss = 0.
    weights = args.style_imgs_weights
    for img, img_weight in zip(style_imgs, weights):
        sess.run(net['input'].assign(img))
        style_loss = 0.
        for layer, weight in zip(args.style_layers, args.style_layer_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x) * weight
        style_loss /= float(len(args.style_layers))
        total_style_loss += (style_loss * img_weight)
    total_style_loss /= float(len(style_imgs))
    return total_style_loss


def sum_content_losses(args, sess, net, content_img):
    sess.run(net['input'].assign(content_img))
    content_loss = 0.
    for layer, weight in zip(args.content_layers, args.content_layer_weights):
        p = sess.run(net[layer])
        x = net[layer]
        p = tf.convert_to_tensor(p)
    content_loss += content_layer_loss(args, p, x) * weight
    content_loss /= float(len(args.content_layers))
    return content_loss


def minimize_with_lbfgs(sess, net, optimizer, init_img):
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)


def minimize_with_adam(args, sess, net, optimizer, init_img, loss):
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < args.max_iterations):
        sess.run(train_op)
        if iterations % args.print_iterations == 0 and args.verbose:
            curr_loss = loss.eval()
            print("At iterate {}\tf=  {:.5E}".format(iterations, curr_loss))
        iterations += 1


def stylize(args, content_img, style_imgs, init_img):
    with tf.device(args.device), tf.Session() as sess:
        # setup network
        net = build_model(args, content_img)

        L_style = sum_style_losses(args, sess, net, style_imgs)

        # content loss
        L_content = sum_content_losses(args, sess, net, content_img)

        # denoising loss
        L_tv = tf.image.total_variation(net['input'])

        # loss weights
        alpha = args.content_weight
        beta = args.style_weight
        theta = args.tv_weight

        # total loss
        L_total = alpha * L_content
        L_total += beta * L_style
        L_total += theta * L_tv

        # optimization algorithm
        optimizer = get_optimizer(args, L_total)

        if args.optimizer == 'adam':
            minimize_with_adam(args, sess, net, optimizer, init_img, L_total)
        elif args.optimizer == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimizer, init_img)

        output_img = sess.run(net['input'])
        return output_img
