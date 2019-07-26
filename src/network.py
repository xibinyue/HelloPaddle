import paddle.fluid as fluid
import paddle


def paddlenet(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')  # 这里是Paddle官方封装的一个组合层，可以定义卷积参数，包括分组卷积、pooling参数、是否使用BN和Dropout 我们直接拿来用

    conv1 = conv_block(input, 32, 1, [0.3])
    conv2 = conv_block(conv1, 64, 1, [0.4])
    conv3 = conv_block(conv2, 128, 1, [0.4])  # 这里不分组，只许设定一个drop比例 如果用分组的话 需要设置多个
    fc1 = fluid.layers.fc(input=conv3, size=256, act=None)
    fc2 = fluid.layers.fc(input=fc1, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict