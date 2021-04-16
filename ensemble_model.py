import numpy as np
import tensorflow as tf
from noise_mnist_utils import normal_parse_params, rec_log_prob
layers = tf.keras.layers

tf.enable_eager_execution()


class ResBlock(tf.keras.Model):
    """
    Usual full pre-activation ResNet bottleneck block.
    """
    def __init__(self, outer_dim, inner_dim):
        super(ResBlock, self).__init__()
        data_format = 'channels_last'

        self.net = tf.keras.Sequential([
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(inner_dim, (1, 1)),
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(inner_dim, (3, 3), padding='same'),
            layers.BatchNormalization(axis=-1),
            layers.LeakyReLU(),
            layers.Conv2D(outer_dim, (1, 1))])

    def call(self, x):
        return x + self.net(x)


class MLPBlock(tf.keras.Model):
    def __init__(self, inner_dim):
        super(MLPBlock, self).__init__()
        self.net = tf.keras.Sequential([
                layers.BatchNormalization(),
                layers.LeakyReLU(),
                layers.Conv2D(inner_dim, (1, 1))])

    def call(self, x):
        return x + self.net(x)


class EncoderNetwork(tf.keras.Model):
    def __init__(self):
        super(EncoderNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8)])

        self.net2 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8)])

        self.net3 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16)])
        self.pad3 = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))

        self.net4 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32)])

        self.net5 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64)])

        self.net6 = tf.keras.Sequential([layers.AveragePooling2D(2, 2), layers.Conv2D(128, 1),
            MLPBlock(128), MLPBlock(128), MLPBlock(128), MLPBlock(128)])

    def call(self, x):
        # 当输入是 (None, 28, 28, 2),
        x = self.net1(x)           # (b, 28, 28, 8)
        x = self.net2(x)           # (b, 14, 14, 16)
        x = self.net3(x)           # (b, 7, 7, 32)
        x = self.pad3(x)           # (b, 8, 8, 32)
        x = self.net4(x)           # (b, 4, 4, 64)
        x = self.net5(x)           # (b, 2, 2, 128)
        x = self.net6(x)           # (b, 1, 1, 128)
        return x


class DecoderNetwork(tf.keras.Model):
    def __init__(self):
        super(DecoderNetwork, self).__init__()
        self.net1 = tf.keras.Sequential([layers.Conv2D(128, 1),
            MLPBlock(128), MLPBlock(128), MLPBlock(128), MLPBlock(128),
            layers.Conv2D(128, 1), layers.UpSampling2D((2, 2))])

        self.net2 = tf.keras.Sequential([layers.Conv2D(128, 1),
            ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64), ResBlock(128, 64),
            layers.Conv2D(64, 1), layers.UpSampling2D((2, 2))])

        self.net3 = tf.keras.Sequential([layers.Conv2D(64, 1),
            ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32), ResBlock(64, 32),
            layers.Conv2D(32, 1), layers.UpSampling2D((2, 2))])

        self.net4 = tf.keras.Sequential([layers.Conv2D(32, 1),
            ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16), ResBlock(32, 16),
            layers.Conv2D(16, 1), layers.UpSampling2D((2, 2))])

        self.net5 = tf.keras.Sequential([layers.Conv2D(16, 1),
            ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8), ResBlock(16, 8),
            layers.Conv2D(8, 1), layers.UpSampling2D((2, 2))])

        self.net6 = tf.keras.Sequential([layers.Conv2D(8, 1),
            ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8), ResBlock(8, 8),
            layers.Conv2D(4, 1)])

        self.net7 = tf.keras.Sequential([layers.Conv2D(2, 1),
            ResBlock(2, 2), ResBlock(2, 2), ResBlock(2, 2),
            layers.Conv2D(2, 1)])

    def call(self, x):             # input=(b, 1, 1, 128)
        x = self.net1(x)           # (b, 2, 2, 128)
        x = self.net2(x)           # (b, 4, 4, 64)
        x = self.net3(x)           # (b, 8, 8, 32)
        x = x[:, :-1, :-1, :]      # (b, 7, 7, 32)
        x = self.net4(x)           # (b, 14, 14, 16)
        x = self.net5(x)           # (b, 28, 28, 8)
        x = self.net6(x)           # (b, 28, 28, 4)
        x = self.net7(x)           # (b, 28, 28, 2)
        return x


class BaselineModel(tf.keras.Model):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.encoder_network = EncoderNetwork()
        self.decoder_network = DecoderNetwork()

    def call(self, x):
        en = self.encoder_network(x)
        de = self.decoder_network(en)
        return de


if __name__ == '__main__':
    encoder_network = EncoderNetwork()
    x1 = tf.convert_to_tensor(np.random.random((2, 28, 28, 2)), tf.float32)
    y2 = encoder_network(x1)
    print("output of encoder network:", y2.shape)

    decoder_network = DecoderNetwork()
    x2 = tf.convert_to_tensor(np.random.random((2, 1, 1, 128)), tf.float32)
    y2 = decoder_network(x2)
    print("output of decoder network:", y2.shape)

    baseline_model = BaselineModel()
    x3 = tf.convert_to_tensor(np.random.random((2, 28, 28, 1)), tf.float32)
    y3 = baseline_model(x3)
    print("output of baseline networks:", y3.shape)

    print("Parameters:", np.sum([np.prod(v.shape.as_list()) for v in encoder_network.trainable_variables]))
    print("Parameters:", np.sum([np.prod(v.shape.as_list()) for v in decoder_network.trainable_variables]))
    print("Total Para:", np.sum([np.prod(v.shape.as_list()) for v in baseline_model.trainable_variables]))

    rec_params = baseline_model(x3)
    rec_loss = -1.0 * rec_log_prob(rec_params=rec_params, s_next=x3)
    print("rec_loss:", rec_loss.shape)
    loss = tf.reduce_mean(rec_loss)
    print("loss:", loss)
    



