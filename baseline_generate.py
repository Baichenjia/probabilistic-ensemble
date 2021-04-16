import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from ensemble_model import BaselineModel
from noise_mnist_utils import normal_parse_params, rec_log_prob

tfd = tfp.distributions
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class MnistBaselineTest(tf.keras.Model):
    def __init__(self, ensemble_num=3):
        super().__init__()
        self.ensemble_num = ensemble_num
        self.baseline_model = [BaselineModel() for _ in range(ensemble_num)]

    def ensemble_loss(self, obs, out_obs):
        total_loss = []
        for idx in range(self.ensemble_num):
            single_loss = self.single_loss(tf.convert_to_tensor(obs), tf.convert_to_tensor(out_obs), idx)
            total_loss.append(single_loss * np.random.random())
        return total_loss

    def single_loss(self, obs, out_obs, i=0):
        """ 输出 variational lower bound, 训练目标是最大化该值. 输出维度 (batch,)
        """
        rec_params = self.baseline_model[i](obs)
        rec_loss = -1.0 * rec_log_prob(rec_params=rec_params, s_next=out_obs)
        loss = tf.reduce_mean(rec_loss)
        return loss

    def generate_samples_params(self, obs):
        """ k 代表采样的个数. 从 prior network 输出分布中采样, 随后输入到 generative network 中采样
        """
        samples = []
        for idx in range(self.ensemble_num):
            sample_params = self.baseline_model[idx](obs)    # (batch,28,28,1)
            samples.append(sample_params[..., 0:1])          # take the mean
        return samples


def build_test_dataset():
    # data
    (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    test_images = np.expand_dims((test_images / 255.).astype(np.float32), axis=-1)
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    image_dict = {}
    # dict of image and label
    for idx in range(len(test_labels)):
        label = test_labels[idx]
        if label not in image_dict.keys():
            image_dict[label] = []
        else:
            image_dict[label].append(idx)

    # 随机选择
    idx0_random = np.random.choice(image_dict[0])    # 抽取数字 0 的所有序号
    idx1_random = np.random.choice(image_dict[1])    # 抽取数字 1 的所有序号

    test_x0 = test_images[idx0_random]    # 转为图像
    test_x1 = test_images[idx1_random]    # 转为图像

    return np.expand_dims(test_x0, axis=0), np.expand_dims(test_x1, axis=0)  # shape=(1,28,28,1)


def generate_0(model):
    test_x, _ = build_test_dataset()   # 取到数字0

    # sample
    samples = model.generate_samples_params(test_x)
    print([s.shape.as_list() for s in samples])

    # plot
    plt.figure(figsize=(10, 10))
    plt.subplot(1, len(samples) + 1, 1)
    plt.axis('off')
    plt.imshow(test_x[0, :, :, 0], cmap='gray')
    plt.title("input", fontsize=20)

    idx = 1
    for sample in samples:
        sample = tf.nn.sigmoid(sample).numpy()
        # sample[sample >= 0.0] = 1.
        # sample[sample < 0.0] = 0.
        assert sample.shape == (1, 28, 28, 1)
        plt.subplot(1, len(samples)+1, idx+1)
        plt.axis('off')
        plt.imshow(sample[0, :, :, 0], cmap='gray')
        plt.title("model "+str(idx), fontsize=20)
        # plt.subplots_adjust(wspace=0., hspace=0.1)
        idx += 1
    plt.savefig("baseline_model/Mnist-Ensemble-res0.pdf")
    # plt.show()
    plt.close()


def generate_1(model):
    _, test_x = build_test_dataset()   # 取到数字0

    # sample
    samples = model.generate_samples_params(test_x)
    print([s.shape.as_list() for s in samples])

    # plot
    plt.figure(figsize=(10, 10))
    plt.subplot(1, len(samples) + 1, 1)
    plt.axis('off')
    plt.imshow(test_x[0, :, :, 0], cmap='gray')
    plt.title("input", fontsize=20)

    idx = 1
    for sample in samples:
        sample = tf.nn.sigmoid(sample).numpy()
        # sample[sample >= 0.0] = 1.
        # sample[sample < 0.0] = 0.
        assert sample.shape == (1, 28, 28, 1)
        plt.subplot(1, len(samples)+1, idx+1)
        plt.axis('off')
        plt.imshow(sample[0, :, :, 0], cmap='gray')
        plt.title("model "+str(idx), fontsize=20)
        # plt.subplots_adjust(wspace=0., hspace=0.1)
        idx += 1
    plt.savefig("baseline_model/Mnist-Ensemble-res1.pdf")
    # plt.show()
    plt.close()


if __name__ == '__main__':
    # initialize model and load weights
    test_x0, test_x1 = build_test_dataset()
    ensemble_model = MnistBaselineTest()
    ensemble_model.ensemble_loss(tf.convert_to_tensor(test_x0), tf.convert_to_tensor(test_x0))
    print("load weights...")
    ensemble_model.load_weights("baseline_model/model.h5")
    print("load done")

    # generate 0
    print("Generate number 0")
    generate_0(ensemble_model)

    # generate 1
    print("Generate number 1")
    generate_1(ensemble_model)
