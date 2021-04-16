import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from ensemble_model import BaselineModel
from noise_mnist_utils import normal_parse_params, rec_log_prob
import pickle

tfd = tfp.distributions
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


class MnistBaseline(tf.keras.Model):
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
        # return tf.convert_to_tensor(total_loss)

    def single_loss(self, obs, out_obs, i=0):
        """ 输出 variational lower bound, 训练目标是最大化该值. 输出维度 (batch,)
        """
        rec_params = self.baseline_model[i](obs)
        rec_loss = -1.0 * rec_log_prob(rec_params=rec_params, s_next=out_obs)
        loss = tf.reduce_mean(rec_loss)
        return loss


def build_dataset(train_images, train_labels, storage0=5, storage1=10):
    image_dict = {}
    # dict of image and label
    for idx in range(len(train_labels)):
        label = train_labels[idx]
        if label not in image_dict.keys():
            image_dict[label] = []
        else:
            image_dict[label].append(idx)

    # 构造数字0的样本
    obs_idx0 = image_dict[0]           # 抽取数字 0 的所有序号
    np.random.shuffle(obs_idx0)
    train_x0, train_y0 = [], []
    for idx in obs_idx0:
        for i in range(storage0):
            train_x0.append(idx)
            trans_to_idx = np.random.choice(image_dict[1])
            train_y0.append(trans_to_idx)
    print("training data x0:", len(train_x0))
    print("training data y0:", len(train_y0))

    # 构造数字1的样本
    obs_idx1 = image_dict[1]           # 抽取数字 1 的所有序号
    np.random.shuffle(obs_idx1)
    train_x1, train_y1 = [], []
    for idx in obs_idx1:
        for i in range(storage1):
            train_x1.append(idx)
            trans_to_label = np.random.randint(low=2, high=10)
            trans_to_idx = np.random.choice(image_dict[trans_to_label])
            train_y1.append(trans_to_idx)
    print("training data x1:", len(train_x1))
    print("training data y1:", len(train_y1))

    train_x0_img = train_images[train_x0]
    train_y0_img = train_images[train_y0]
    print("\ntraining data x0:", train_x0_img.shape)
    print("training data y0:", train_y0_img.shape)

    train_x1_img = train_images[train_x1]
    train_y1_img = train_images[train_y1]
    print("\ntraining data x1:", train_x1_img.shape)
    print("training data y1:", train_y1_img.shape)

    train_x_img = np.vstack([train_x0_img, train_x1_img])
    train_y_img = np.vstack([train_y0_img, train_y1_img])
    print("\ntraining data x:", train_x_img.shape)
    print("training data y:", train_y_img.shape)
    return train_x_img, train_y_img


def mnist_data(build_train=True):
    # data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = np.expand_dims((train_images / 255.).astype(np.float32), axis=-1)
    test_images = np.expand_dims((test_images / 255.).astype(np.float32), axis=-1)

    # Binarization
    train_images[train_images >= .5] = 1.
    train_images[train_images < .5] = 0.
    test_images[test_images >= .5] = 1.
    test_images[test_images < .5] = 0.

    # train
    if build_train:
        print("Generating training data:")
        train_x, train_y = build_dataset(train_images, train_labels, storage0=5, storage1=50)
        np.save('data/train_x.npy', train_x) 
        np.save('data/train_y.npy', train_y) 
    else:
        train_dataset = None

    print("Generating testing data:")
    test_x, test_y = build_dataset(test_images, test_labels, storage0=5, storage1=10)
    np.save('data/test_x.npy', test_x) 
    np.save('data/test_y.npy', test_y) 
    print("dataset done.")


def load_mnist_data():
    train_x = np.load("data/train_x.npy")
    train_y = np.load("data/train_y.npy")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(500000)
    train_dataset = train_dataset.batch(512, drop_remainder=True)
        
    test_x = tf.convert_to_tensor(np.load("data/test_x.npy"))
    test_y = tf.convert_to_tensor(np.load("data/test_y.npy"))
    
    return train_dataset, test_x, test_y



def train():
    # model
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
    mnist_baseline = MnistBaseline()

    # data
    # mnist_data(build_train=True)  # 先 run 这个来保存到本地
    train_dataset, test_x, test_y = load_mnist_data()

    # start train
    Epochs = 500
    test_loss = []
    for epoch in range(Epochs):
        print("Epoch: ", epoch)
        for i, (batch_x, batch_y) in enumerate(train_dataset):
            with tf.GradientTape() as tape:                          # train
                loss_ensemble = mnist_baseline.ensemble_loss(batch_x, batch_y)
                loss = tf.reduce_mean(loss_ensemble)
                if i % 10 == 0:
                    print(i, ", loss_ensemble:", [x.numpy() for x in loss_ensemble], ", loss:", loss.numpy(), flush=True)
            gradients = tape.gradient(loss, mnist_baseline.trainable_variables)
            # gradients, _ = tf.clip_by_global_norm(gradients, 1.)
            optimizer.apply_gradients(zip(gradients, mnist_baseline.trainable_variables))

        # test
        t_loss = tf.reduce_mean(mnist_baseline.ensemble_loss(test_x, test_y))
        test_loss.append(t_loss)
        print("Test Loss:", t_loss)

        # save
        mnist_baseline.save_weights("baseline_model/model_"+str(epoch)+".h5")
        np.save("baseline_model/test_loss.npy", np.array(test_loss))


if __name__ == '__main__':
    train()

