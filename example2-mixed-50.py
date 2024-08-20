import argparse
import numpy as np
import matplotlib
from tqdm import tqdm
import copy

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import ot
import ot.plot
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,3"  # specify which GPU(s) to be used


def get_opt(x,list_value=None,list_i=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--X_number', type=int, default=x, help='the estimated X')  ##yaogai##
    parser.add_argument('--list_numbers', type=list, default=[1,2,3,4,5,6,7,8,9,10], help='which dataset to use for list_value')  ##yaogai##

    parser.add_argument('--DATASET_X', type=str,default=f'模拟次数{list_i}-list值{list_value}-mixedexample2-50(x={x})')  ##yaogai##
    # parser.add_argument('--save_file_name', type=str,default=f'模拟次数{list_i}-list值{list_value}-mixedexample2-50(x={x})')

    parser.add_argument('--list_value', type=int, default=list_value)  # list的值
    parser.add_argument('--list_i', type=int, default=list_i)  # 模拟次数

    parser.add_argument('--path_name',type= str,
                        default=f'模拟次数{list_i}-list值{list_value}-mixedexample2-50(x={x})'.split('50')[0][:-1].replace(f'list值{list_value}', f'list值{list_value - 1}'))

    parser.add_argument('--DATASET_Y', type=str, default='spherical uniform', help='which dataset to use for Y')

    parser.add_argument('--SHOW_THE_PLOT', type=bool, default=False, help='Boolean option to show the plots or not')
    parser.add_argument('--DRAW_THE_ARROWS', type=bool, default=False, help='Whether to draw transport arrows or not')

    parser.add_argument('--TRIAL', type=int, default=1, help='the trail no.')

    parser.add_argument('--LAMBDA', type=float, default=1, help='Regularization constant for positive weight constraints')

    parser.add_argument('--NUM_NEURON', type=int, default=32, help='number of neurons per layer')

    parser.add_argument('--NUM_LAYERS', type=int, default=4, help='number of hidden layers before output')

    parser.add_argument('--LR', type=float, default=1e-4, help='learning rate')

    parser.add_argument('--ITERS', type=int, default=10000, help='number of iterations of training')

    parser.add_argument('--BATCH_SIZE', type=int, default=1025, help='size of the batches')

    parser.add_argument('--SCALE', type=float, default=5.0, help='scale for the gaussian_mixtures')
    parser.add_argument('--VARIANCE', type=float, default=0.5, help='variance for each mixture')

    parser.add_argument('--N_TEST', type=int, default=101, help='number of test samples')
    parser.add_argument('--N_PLOT', type=int, default=512, help='number of samples for plotting')
    parser.add_argument('--N_CPU', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--INPUT_DIM', type=int, default=2, help='dimensionality of the input x')
    parser.add_argument('--N_GENERATOR_ITERS', type=int, default=10,
                        help='number of training steps for discriminator per iter')

    return parser.parse_args()


def main(opt):
    # specify the convex function class
    print("specify the convex function class")

    hidden_size_list = [opt.NUM_NEURON for i in range(opt.NUM_LAYERS)]

    hidden_size_list.append(1)

    print(hidden_size_list)

    fn_model = Kantorovich_Potential(opt.INPUT_DIM, hidden_size_list)
    gn_model = Kantorovich_Potential(opt.INPUT_DIM, hidden_size_list)

    SET_PARAMS_NAME = str(opt.BATCH_SIZE) + '_batch' + str(opt.NUM_NEURON) + '_neurons' + str(
        opt.NUM_LAYERS) + '_layers'
    EXPERIMENT_NAME = opt.DATASET_X

    # Directory to store the images

    os.makedirs("figures/{0}/".format(EXPERIMENT_NAME,), exist_ok=True)

    # Define the test set
    print("Define the test set")
    X_test = next(sample_data_gen(opt.DATASET_X, opt.N_TEST, opt.SCALE, opt.VARIANCE,opt))

    Y_test = next(sample_data_gen(opt.DATASET_Y, opt.N_TEST, opt.SCALE, opt.VARIANCE,opt))

    ##Define the tauS(d-1)
    BATCH_SIZE = 1025
    r = int(np.sqrt(BATCH_SIZE - 1))
    radium = np.linspace(0 + 1 / r, 1 + 1 / r, r, endpoint=True)
    angle = 2 * np.pi * np.linspace(0, 99 / 100,100)

    tauS_list = []
    tautau = [0.01, 0.04, 0.07, 0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.30, 0.32,
              0.35, 0.38, 0.41, 0.44, 0.47, 0.5, 0.54, 0.57, 0.6, 0.63, 0.66, 0.70,
              0.72, 0.75, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97]

    for i in range(0, 32):
        tau = tautau[i]
        tauS = np.array([radium[int(r * tau)] * np.cos(angle), radium[int(r * tau)] * np.sin(angle)])
        tauS_list.append(tauS)
    median = np.array([[0] * len(angle), [0] * len(angle)]).T

    saver = tf.train.Saver()

    # Running the optimization
    with tf.Session() as sess:
        compute_OT = ComputeOT(sess, opt.INPUT_DIM, fn_model, gn_model, opt.LR,opt)  # initilizing

        compute_OT.learn(opt.BATCH_SIZE, opt.ITERS, opt.N_GENERATOR_ITERS, opt.SCALE, opt.VARIANCE, opt.DATASET_X,
                         opt.DATASET_Y, opt.N_PLOT, EXPERIMENT_NAME, opt)  # learning the optimal map

        tauS_jieguo = {}
        for i, itme in enumerate(tauS_list):
            jieguo = compute_OT.transport_Y_to_X(itme.T)
            tauS_jieguo['tauspred{}_1'.format(i)] = jieguo[:, 0]
            tauS_jieguo['tauspred{}_2'.format(i)] = jieguo[:, 1]

        medianpred = compute_OT.transport_Y_to_X(median)
        tauS_jieguo['medianpred_1'] = medianpred[:, 0]
        tauS_jieguo['medianpred_2'] = medianpred[:, 1]
        tauS_jieguo_df = pd.DataFrame(tauS_jieguo)

        # tauS_jieguo_df.to_excel(r'./11-itertianbu1-mixedexample2-50(x={}).xlsx'.format(opt.X_number))
        tauS_jieguo_df.to_excel(opt.DATASET_X + '.xlsx')

        save_path = "saving_model/{0}".format(EXPERIMENT_NAME)
        # # 出现昨晚的BUG就解开这边两行代码
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        #
        # saver.save(sess,  save_path + '/model-{}.ckpt'.format(SET_PARAMS_NAME + str(opt.ITERS) + '_iters'))  # 这里的保存路径 没有就创建

        print("Final Wasserstein distance: {0}".format(compute_OT.compute_W2(X_test, Y_test)))

    # Using exact OT solvers in Python
    print("Actual Wasserstein distance: {0}".format(python_OT(X_test, Y_test, opt.N_TEST)))

class ComputeOT:

    def __init__(self, sess, input_dim, f_model, g_model, lr,opt):

        self.sess = sess

        self.f_model = f_model
        self.g_model = g_model

        self.input_dim = input_dim

        self.x = tf.placeholder(tf.float32, [None, input_dim])
        self.y = tf.placeholder(tf.float32, [None, input_dim])

        self.fx = self.f_model.forward(self.x)

        self.gy = self.g_model.forward(self.y)

        [self.grad_fx] = tf.gradients(self.fx, self.x)

        [self.grad_gy] = tf.gradients(self.gy, self.y)

        self.f_grad_gy = self.f_model.forward(self.grad_gy)
        self.y_dot_grad_gy = tf.reduce_sum(tf.multiply(self.y, self.grad_gy), axis=1, keepdims=True)

        self.x_squared = tf.reduce_sum(tf.multiply(self.x, self.x), axis=1, keepdims=True)
        self.y_squared = tf.reduce_sum(tf.multiply(self.y, self.y), axis=1, keepdims=True)

        self.f_loss = tf.reduce_mean(self.fx - self.f_grad_gy)
        self.g_loss = tf.reduce_mean(self.f_grad_gy - self.y_dot_grad_gy)

        self.f_postive_constraint_loss = self.f_model.positive_constraint_loss
        self.g_postive_constraint_loss = self.g_model.positive_constraint_loss

        if opt.LAMBDA > 0:

            self.f_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(self.f_loss,
                                                                                                       var_list=self.f_model.var_list)
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.9).minimize(
                self.g_loss + opt.LAMBDA * self.g_postive_constraint_loss, var_list=self.g_model.var_list)

        else:

            self.f_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.f_loss,
                                                                                 var_list=self.f_model.var_list)
            self.g_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.g_loss,
                                                                                 var_list=self.g_model.var_list)

        self.W2 = tf.reduce_mean(
            self.f_grad_gy - self.fx - self.y_dot_grad_gy + 0.5 * self.x_squared + 0.5 * self.y_squared)

        self.init = tf.global_variables_initializer()

    def learn(self, batch_size, iters, inner_loop_iterations, scale, variance, dataset_x, dataset_y, plot_size,
              experiment_name, opt):

        print_T = 10

        save_figure_iterations = 1000

        self.sess.run(self.init)

        '''
        dataset_x 这个值弄成列表就行了
        '''

        data_gen_x = sample_data_gen(dataset_x, batch_size, scale, variance,opt)
        print("data_gen_x created")
        data_gen_y = sample_data_gen(dataset_y, batch_size, scale, variance,opt)
        print("data_gen_y created")

        # This data will be used for plotting
        X_plot = next(sample_data_gen(dataset_x, plot_size, scale, variance,opt))
        Y_plot = next(sample_data_gen(dataset_y, plot_size, scale, variance,opt))
        print("Plotting data created")

        if opt.LAMBDA > 0:
            trainable_g_list = [self.g_optimizer]
            trainable_f_list = [self.f_optimizer, self.f_model.proj]

        else:
            trainable_g_list = [self.g_optimizer, self.g_model.proj]
            trainable_f_list = [self.f_optimizer, self.f_model.proj]

        for iteration in range(iters):

            for j in range(inner_loop_iterations):
                x_train = next(data_gen_x)
                y_train = next(data_gen_y)

                # Training the g neural network
                _ = self.sess.run(trainable_g_list, feed_dict={self.x: x_train, self.y: y_train})

            x_train = next(data_gen_x)
            y_train = next(data_gen_y)

            # Training the f_neural network
            _ = self.sess.run(trainable_f_list, feed_dict={self.x: x_train, self.y: y_train})

            if iteration % print_T == 0:
                f_loss, g_loss, W2 = self.sess.run([self.f_loss, self.g_loss, self.W2],
                                                   feed_dict={self.x: x_train, self.y: y_train})
                print("Iterations = %i, f_loss = %.4f, g_loss = %.4f, W2 = %.4f" % (iteration, f_loss, g_loss, W2))

            if (iteration + 1) % save_figure_iterations == 0:
                self.save_the_figure(iteration + 1, X_plot, Y_plot, experiment_name, opt)

    def transport_X_to_Y(self, X):

        T_X_to_Y = self.sess.run(self.grad_fx, feed_dict={self.x: X})

        return T_X_to_Y

    def transport_Y_to_X(self, Y):

        T_Y_to_X = self.sess.run(self.grad_gy, feed_dict={self.y: Y})

        return T_Y_to_X

    def eval_gy(self, Y):

        _gy = self.sess.run(self.gy, feed_dict={self.y: Y})

        return _gy

    def compute_W2(self, X, Y):

        return self.sess.run(self.W2, feed_dict={self.x: X, self.y: Y})

    def save_the_figure(self, iteration, X_plot, Y_plot, experiment_name, opt):

        (plot_size, _) = np.shape(X_plot)

        X_pred = self.transport_Y_to_X(Y_plot)

        fig = plt.figure()

        plt.scatter(Y_plot[:, 0], Y_plot[:, 1], color='C1',
                    alpha=0.5, label=r'$Y$')
        plt.scatter(X_plot[:, 0], X_plot[:, 1], color='C2',
                    alpha=0.5, label=r'$X$')
        plt.scatter(X_pred[:, 0], X_pred[:, 1], color='C3',
                    alpha=0.5, label=r'$\nabla g(Y)$')

        plt.legend()

        if opt.DRAW_THE_ARROWS:

            for i in range(plot_size):
                drawArrow(Y_plot[i, :], X_pred[i, :])

        if opt.SHOW_THE_PLOT:
            plt.show()

        fig.savefig("figures/{0}/{1}.png"
                    .format(experiment_name, str(iteration)))

        print("Plot saved at iteration {0}".format(iteration))


class Kantorovich_Potential:
    '''
        Modelling the Kantorovich potential as Input convex neural network (ICNN)
        input: y
        output: z = h_L
        Architecture: h_1     = ReLU^2(A_0 y + b_0)
                      h_{l+1} =   ReLU(A_l y + b_l + W_{l-1} h_l)
        Constraint: W_l > 0
    '''

    def __init__(self, input_size, hidden_size_list):

        # hidden_size_list always contains 1 in the end because it's a scalar output
        self.input_size = input_size
        self.num_hidden_layers = len(hidden_size_list)

        # list of matrices that interacts with input
        self.A = []
        for k in range(0, self.num_hidden_layers):
            self.A.append(
                tf.Variable(tf.random_uniform([self.input_size, hidden_size_list[k]], maxval=0.1), dtype=tf.float32))

        # list of bias vectors at each hidden layer
        self.b = []
        for k in range(0, self.num_hidden_layers):
            self.b.append(tf.Variable(tf.zeros([1, hidden_size_list[k]]), dtype=tf.float32))

        # list of matrices between consecutive layers
        self.W = []
        for k in range(1, self.num_hidden_layers):
            self.W.append(tf.Variable(tf.random_uniform([hidden_size_list[k - 1], hidden_size_list[k]], maxval=0.1),
                                      dtype=tf.float32))

        self.var_list = self.A + self.b + self.W

        self.positive_constraint_loss = tf.add_n([tf.nn.l2_loss(tf.nn.relu(-w)) for w in self.W])

        self.proj = [w.assign(tf.nn.relu(w)) for w in self.W]  # ensuring the weights to stay positive

    def forward(self, input_y):

        # Using ReLU Squared
        z = tf.nn.leaky_relu(tf.matmul(input_y, self.A[0]) + self.b[0], alpha=0.2)
        z = tf.multiply(z, z)

        # # If we want to use ReLU and softplus for the input layer
        # z = tf.matmul(input_y, self.A[0]) + self.b[0]
        # z = tf.multiply(tf.nn.relu(z),tf.nn.softplus(z))

        # If we want to use the exponential layer for the input layer
        ## z=tf.nn.softplus(tf.matmul(input_y, self.A[0]) + self.b[0])

        for k in range(1, self.num_hidden_layers):
            z = tf.nn.leaky_relu(tf.matmul(input_y, self.A[k]) + self.b[k] + tf.matmul(z, self.W[k - 1]))

        return z


def sample_data_gen(DATASET, BATCH_SIZE, SCALE, VARIANCE,opt):  # 这里
    print('Dataset:',DATASET)
    '''
    while True:

        centers = [np.array([1.,0.]), np.array([-1.,0.]), np.array([0., 1.]), np.array([0.,-1.])]

        dataset = []
        for i in range(batch_size):
            point = np.random.randn(2)*variance
            center = scale*random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        #dataset /= 1.414 # stdev
        yield dataset
    '''
    if DATASET == '25gaussians':

        dataset = []
        for i in range(100000 / 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        # dataset /= 2.828 # stdev
        while True:
            for i in range(len(dataset) / BATCH_SIZE):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'spherical uniform':
        r = int(np.sqrt(BATCH_SIZE - 1))
        radium = np.linspace(0 + 1 / r, 1 + 1 / r, r, endpoint=True)
        angle = 2 * np.pi * np.linspace(0, 1 - 1 / r, r, endpoint=True)
        name_1 = np.array([radium]).T * np.cos(angle)
        name_2 = np.array([radium]).T * np.sin(angle)
        U = np.array([name_1.reshape(-1), name_2.reshape(-1)])
        U = U.T

        while True:
            dataset = np.zeros((U.shape[0] + 1, 2))
            dataset[:-1, 0] = U[:, 0]
            dataset[:-1, 1] = U[:, 1]
            yield dataset

    # elif DATASET == opt.DATASET_X and opt.list_value == 0:##yaogai##  # list = 0  # 判断使用那种数据生成方式的
    elif opt.list_value == 0:  # list = 0  # 判断使用那种数据生成方式的
        print('此处进入到了list = 0')
        print('list value:' ,opt.list_value)
        print('------------------')
        while True:
            sam = 10000
            xx = 4 * np.random.random(sam) - 2
            x = opt.X_number  ####X=x##yaogai##
            Id = np.array([[1, 0], [0, 1]])
            et = ot.datasets.make_2D_samples_gauss(sam, [0, 0], Id)
            y = xx + (1 + (np.sin(np.pi * xx / 2)) ** 2 * (3 / 2)) * et[:, 0]
            z = xx ** 2 + (1 + (np.sin(np.pi * xx / 2)) ** 2 * (3 / 2)) * et[:, 1]

            c1 = np.random.normal(-3, 2, sam)
            c2 = np.random.exponential(1 / 0.22, sam)
            c3 = np.random.normal(-1.8, 2, sam)
            c4 = np.random.exponential(1 / 0.1, sam)
            ycen = copy.deepcopy(y)
            zcen = copy.deepcopy(z)
            Ind1 = np.ones(sam)  # (YTREU,ZTRUE,Ind1,Ind2)
            Ind2 = np.ones(sam)  # (YTREU,ZTRUE,Ind1,Ind2)
            Y_index_2 = np.where(y < c1)
            ycen[Y_index_2] = c1[Y_index_2]
            Ind1[Y_index_2] = 2 ##2为左删失
            Y_index_3 = np.where(y > c2)
            ycen[Y_index_3] = c2[Y_index_3]
            Ind1[Y_index_3] = 3 ##3为右删失
            Z_index_2 = np.where(z < c3)
            zcen[Z_index_2] = c3[Z_index_2]
            Ind2[Z_index_2] = 2
            Z_index_3 = np.where(z > c4)
            zcen[Z_index_3] = c4[Z_index_3]
            Ind2[Z_index_3] = 3

            tmp_dataset = np.zeros((sam, 5))  # (YTREU,ZTRUE,Ind1,Ind2)
            tmp_dataset[:, 0] = xx
            tmp_dataset[:, 1] = ycen
            tmp_dataset[:, 2] = zcen
            tmp_dataset[:, 3] = Ind1
            tmp_dataset[:, 4] = Ind2
            tmp_dataset_index = np.where(tmp_dataset[:, 3] + tmp_dataset[:, 4] == 2)
            tmp = tmp_dataset[tmp_dataset_index]

            k = BATCH_SIZE
            ###Nearest neighbourhood
            order = np.argsort(np.abs(tmp[:,0] - x))
            Y = tmp[:,1][order][:k]
            Z = tmp[:,2][order][:k]

            dataset = np.array([Y,Z]).T
            yield dataset

    # elif DATASET == opt.save_file_name and opt.list_value in opt.list_numbers:##yaogai##
    elif opt.list_value in opt.list_numbers:
        print('list value:', opt.list_value)
        print('=================')
        path_name = opt.path_name
        imputa1 = pd.read_excel('./' + path_name + '-50(x={}).xlsx'.format(opt.X_number), index_col=0)

        tmp_imputa1 = imputa1[imputa1.columns[:-2]]
        # tmp_imputa1 = tmptmp_imputa1[tmptmp_imputa1.columns[36:, ]]
        y1_imputa1 = []
        y2_imputa1 = []
        for l, item in enumerate(tmp_imputa1.values.T):
            if l % 2 == 0:
                y1_imputa1.extend(item)
            else:
                y2_imputa1.extend(item)
        y1_imputa1 = np.array(y1_imputa1)
        y2_imputa1 = np.array(y2_imputa1)

        while True:
            x = opt.X_number
            sam = 10000
            xx = 4 * np.random.random(sam) - 2
            Id = np.array([[1, 0], [0, 1]])
            et = ot.datasets.make_2D_samples_gauss(sam, [0, 0], Id)
            y = xx + (1 + (np.sin(np.pi * xx / 2)) ** 2 * (3 / 2)) * et[:, 0]
            z = xx ** 2 + (1 + (np.sin(np.pi * xx / 2)) ** 2 * (3 / 2)) * et[:, 1]

            c1 = np.random.normal(-3, 2, sam)
            c2 = np.random.exponential(1 / 0.22, sam)
            c3 = np.random.normal(-1.8, 2, sam)
            c4 = np.random.exponential(1 / 0.1, sam)
            ycen = copy.deepcopy(y)
            zcen = copy.deepcopy(z)
            Ind1 = np.ones(sam)  # (YTREU,ZTRUE,Ind1,Ind2)
            Ind2 = np.ones(sam)  # (YTREU,ZTRUE,Ind1,Ind2)
            Y_index_2 = np.where(y < c1)
            ycen[Y_index_2] = c1[Y_index_2]
            Ind1[Y_index_2] = 2 ##2为左删失
            Y_index_3 = np.where(y > c2)
            ycen[Y_index_3] = c2[Y_index_3]
            Ind1[Y_index_3] = 3 ##3为右删失
            Z_index_2 = np.where(z < c3)
            zcen[Z_index_2] = c3[Z_index_2]
            Ind2[Z_index_2] = 2
            Z_index_3 = np.where(z > c4)
            zcen[Z_index_3] = c4[Z_index_3]
            Ind2[Z_index_3] = 3

            k = BATCH_SIZE
            order = np.argsort(np.abs(xx - x))
            ycen_order = ycen[order][:k]
            zcen_order = zcen[order][:k]
            Ind1_order = Ind1[order][:k]
            Ind2_order = Ind2[order][:k]
            # xx_order = xx[order][:k]

            ytrue = copy.deepcopy(ycen_order)
            ztrue = copy.deepcopy(zcen_order)
            Ind1_1 = set(np.where(Ind1_order == 1)[0])
            Ind1_2 = set(np.where(Ind1_order == 2)[0])
            Ind1_3 = set(np.where(Ind1_order == 3)[0])
            Ind2_1 = set(np.where(Ind2_order == 1)[0])
            Ind2_2 = set(np.where(Ind2_order == 2)[0])
            Ind2_3 = set(np.where(Ind2_order == 3)[0])
            index1and2 = Ind1_1 & Ind2_2
            index1and3 = Ind1_1 & Ind2_3
            index2and1 = Ind1_2 & Ind2_1
            index2and2 = Ind1_2 & Ind2_2
            index2and3 = Ind1_2 & Ind2_3
            index3and1 = Ind1_3 & Ind2_1
            index3and2 = Ind1_3 & Ind2_2
            index3and3 = Ind1_3 & Ind2_3

            for i in index1and2:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy2_index = np.where(y2 < zcen_order[i])
                    yy2 = y2[yy2_index]
                    chouy = random.sample(range(len(yy2)), 1)
                    ztrue[i] = yy2[chouy]
                except:
                    ztrue[i] = zcen_order[i]

            for i in index1and3:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy2_index = np.where(y2 > zcen_order[i])
                    yy2 = y2[yy2_index]
                    chouy = random.sample(range(len(yy2)), 1)
                    ztrue[i] = yy2[chouy]
                except:
                    ztrue[i] = zcen_order[i]

            for i in index2and1:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy1_index = np.where(y1 < ycen_order[i])
                    yy1 = y1[yy1_index]
                    chouy = random.sample(range(len(yy1)), 1)
                    ytrue[i] = yy1[chouy]
                except:
                    ytrue[i] = ycen_order[i]

            for i in index2and2:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy1_index = np.where(y1 < ycen_order[i])
                    yy1 = y1[yy1_index]
                    yy2 = y2[yy1_index]
                    yy2_index = np.where(yy2 < zcen_order[i])
                    yyy1 = yy1[yy2_index]
                    yyy2 = yy2[yy2_index]
                    chouy = random.sample(range(len(yyy1)), 1)
                    ytrue[i] = yyy1[chouy]
                    ztrue[i] = yyy2[chouy]
                except:
                    ytrue[i] = ycen_order[i]
                    ztrue[i] = zcen_order[i]

            for i in index2and3:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy1_index = np.where(y1 < ycen_order[i])
                    yy1 = y1[yy1_index]
                    yy2 = y2[yy1_index]
                    yy2_index = np.where(yy2 > zcen_order[i])
                    yyy1 = yy1[yy2_index]
                    yyy2 = yy2[yy2_index]
                    chouy = random.sample(range(len(yyy1)), 1)
                    ytrue[i] = yyy1[chouy]
                    ztrue[i] = yyy2[chouy]
                except:
                    ytrue[i] = ycen_order[i]
                    ztrue[i] = zcen_order[i]

            for i in index3and1:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy1_index = np.where(y1 > ycen_order[i])
                    yy1 = y1[yy1_index]
                    chouy = random.sample(range(len(yy1)), 1)
                    ytrue[i] = yy1[chouy]
                except:
                    ytrue[i] = ycen_order[i]

            for i in index3and2:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy1_index = np.where(y1 > ycen_order[i])
                    yy1 = y1[yy1_index]
                    yy2 = y2[yy1_index]
                    yy2_index = np.where(yy2 < zcen_order[i])
                    yyy1 = yy1[yy2_index]
                    yyy2 = yy2[yy2_index]
                    chouy = random.sample(range(len(yyy1)), 1)
                    ytrue[i] = yyy1[chouy]
                    ztrue[i] = yyy2[chouy]
                except:
                    ytrue[i] = ycen_order[i]
                    ztrue[i] = zcen_order[i]

            for i in index3and3:
                y1 = y1_imputa1
                y2 = y2_imputa1
                try:
                    yy1_index = np.where(y1 > ycen_order[i])
                    yy1 = y1[yy1_index]
                    yy2 = y2[yy1_index]
                    yy2_index = np.where(yy2 > zcen_order[i])
                    yyy1 = yy1[yy2_index]
                    yyy2 = yy2[yy2_index]
                    chouy = random.sample(range(len(yyy1)), 1)
                    ytrue[i] = yyy1[chouy]
                    ztrue[i] = yyy2[chouy]
                except:
                    ytrue[i] = ycen_order[i]
                    ztrue[i] = zcen_order[i]

            dataset = np.array([ytrue, ztrue]).T
            yield dataset


def python_OT(X, Y, n):
    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

    # loss matrix
    M = ot.dist(Y, X)
    scaling = M.max()
    M /= M.max()

    G0 = ot.emd(a, b, M)

    return 0.5 * scaling * sum(sum(G0 * M))  # 0.5 to account for the half quadratic cost


def generate_uniform_around_centers(centers, variance):
    num_center = len(centers)

    return centers[np.random.choice(num_center)] + variance * np.random.uniform(-1, 1, (2))


def generate_cross(centers, variance):
    num_center = len(centers)
    x = variance * np.random.uniform(-1, 1)
    y = (np.random.randint(2) * 2 - 1) * x

    return centers[np.random.choice(num_center)] + [x, y]


def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], color=[0.5, 0.5, 1], alpha=0.3)
    # head_width=0.01, length_includes_head=False)


if __name__ == '__main__':
    x = [0.5]
    my_list=[0,1,2,3,4,5]

    for i in range(1,10):
        for value in tqdm(my_list):
            print(f'当前my_list值位{value}')
            opt = get_opt(round(x[0], 2), list_value=value, list_i=i)  # list_i:意思是第几次模拟。
            main(opt)
