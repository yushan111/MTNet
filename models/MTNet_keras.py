import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dropout, Reshape, Conv2D, GRUCell, Layer, RNN
from tensorflow.keras.initializers import TruncatedNormal, Constant


class AttentionRNN(Layer):

    def __init__(self, hidden_sizes, input_keep_prob, output_keep_prob):
        super(AttentionRNN, self).__init__()
        # tensorflow
        rnns = [tf.nn.rnn_cell.GRUCell(h_size, activation=tf.nn.relu) for h_size in
                hidden_sizes]
        # dropout
        if input_keep_prob < 1 or output_keep_prob < 1:
            rnns = [tf.nn.rnn_cell.DropoutWrapper(rnn,
                                                  input_keep_prob=input_keep_prob,
                                                  output_keep_prob=output_keep_prob)
                    for rnn in rnns]

        if len(rnns) > 1:
            self.rnns = tf.nn.rnn_cell.MultiRNNCell(rnns)
        else:
            self.rnns = rnns[0]
        # keras
        rnn_cells = [GRUCell(h_size, activation="relu", dropout=1-input_keep_prob)
                     for h_size in hidden_sizes]
        # output dropout of GRUCell is omitted since the output_keep_prob is 1 in all configs.
        self.rnns = RNN(rnn_cells)
        self.last_rnn_size = hidden_sizes[-1]

    def build(self, input_shape):
        # input_shape is <batch_size, Tc, hidden_size>
        Tc = input_shape[1]
        # hidden size equals self.last_rnn_size
        hidden_size = input_shape[2]
        self.v = self.add_weight(shape=(Tc, 1),
                                 initializer=TruncatedNormal(stddev=0.1),
                                 name="att_v")
        self.w = self.add_weight(shape=(hidden_size, Tc),
                                 initializer=TruncatedNormal(stddev=0.1),
                                 name="att_w")
        self.u = self.add_weight(shape=(Tc, Tc),
                                 initializer=TruncatedNormal(stddev=0.1),
                                 name="att_u")
        self.rnns.build()
        self.built = True

    def call(self, inputs):







        h_state = states[0]
        s_state = states[1]
        # h(t-1) dot attr_w
        h_part = tf.matmul(h_state, self.w)

        # en_conv_hidden_size * <batch_size_new, 1>
        e_ks = tf.TensorArray(tf.float32, self.last_rnn_size)
        _, output = tf.while_loop(
            lambda i, _: tf.less(i, self.last_rnn_size),
            lambda i, output_ta: (i + 1, output_ta.write(i, tf.matmul(
                tf.tanh(h_part + tf.matmul(inputs[:, i], self.u)), self.v))),
            [0, e_ks])
        # <batch_size, en_conv_hidden_size, 1>
        e_ks = tf.transpose(output.stack(), perm=[1, 0, 2])
        e_ks = tf.reshape(e_ks, shape=[-1, self.last_rnn_size])

        # <batch_size, en_conv_hidden_size>
        a_ks = tf.nn.softmax(e_ks)

        x_t = tf.matmul(tf.expand_dims(inputs[:, :, t], -2), tf.matrix_diag(a_ks))
        # <batch_size, en_conv_hidden_size>
        x_t = tf.reshape(x_t, shape=[-1, self.config.en_conv_hidden_size])

        h_state, s_state = self.rnns(x_t, s_state)
        return h_state, [h_state, s_state]


class MTNet_keras:

    def __init__(self, check_optional_config=True, future_seq_len=2):
        """
        Constructor of MTNet model
        """
        # config parameter
        self.output_dim = future_seq_len
        self.time_step = None  # timestep
        self.cnn_filter = None  # convolution window size (convolution filter height)` ?
        self.long_num = None  # the number of the long-term memory series
        self.ar_size = None  # the window size of ar model
        self.input_dim = None  # input's variable dimension (convolution filter width)
        self.output_dim = None # output's variable dimension
        self.en_conv_hidden_size = None
        # last size is equal to en_conv_hidden_size, should be a list
        self.en_rnn_hidden_sizes = None
        self.input_keep_prob_value = None
        self.output_keep_prob_value = None
        self.lr_value = None

        self.batch_size = None
        self.metric = None
        self.built = 0  # indicating if a new model needs to be build
        self.sess = None  # the session

        # graph component assignment
        # placeholders
        self.X = None
        self.Q = None
        self.Y = None
        self.input_keep_prob = None
        self.output_keep_prob = None
        self.lr = None
        # non-placeholders
        self.y_pred = None
        self.loss = None
        self.train_op = None

        self.check_optional_config = check_optional_config
        self.saved_configs = None

        self.model = None
        self.past_seq_len = None
        self.future_seq_len = future_seq_len
        self.feature_num = None
        self.target_col_num = None
        self.metric = None
        self.latent_dim = None
        self.batch_size = None
        self.check_optional_config = check_optional_config

    def _build_train(self, mc=False, **config):
        """
        build MTNet model
        :param config:
        :return:
        """
        super()._check_config(**config)
        self.config = config
        # long-term time series historical data inputs
        long_input = Input(shape=(self.config["long_num"], self.config["step"],
                                  self.config["feature_num"]))
        # short-term time series historical data
        short_input = Input(shape=(self.config["step"], self.config["feature_num"])

        # ------- no-linear component----------------
        last_rnn_hid_size = self.config["en_rnn_hidden_sizes"][-1]


    def __encoder(self, input, n, name='Encoder'):
        """
            Treat batch_size dimension and num dimension as one batch_size dimension
            (batch_size * num).
        :param input:  <batch_size, num, time_step, input_dim>
        :param n: the number of input time series data. For short term data, the num is 1.
        :return: the embedded of the input <batch_size, num, last_rnn_hid_size>
        """
        name = 'Encoder_' + name
        batch_size_new = self.config["batch_size"] * n
        Tc = self.config["time_step"] - self.config["filter_size"] + 1
        last_rnn_hidden_size = self.config["en_rnn_hidden_sizes"][-1]
        dropout = self.config["dropout"]

        # CNN
        # output: <batch_size_new, conv_out, 1, en_conv_hidden_size>
        reshaped_input = Reshape((self.config["time_step"], self.config["D"], 1))(input)
        cnn_out = Conv2D(filters=self.config["en_conv_hidden_size"],
                         kernel_size=(self.config["ar_size"], self.config["D"]), padding="valid",
                         kernel_initializer=TruncatedNormal(stddev=0.1),
                         bias_initializer=Constant(0.1),
                         activation="relu")(reshaped_input)
        cnn_out = Dropout(dropout)(cnn_out)

        # rnn inputs
        # <batch_size, n, conv_out, en_conv_hidden_size>
        rnn_input = Reshape(cnn_out, shape=[n, Tc, self.config["en_conv_hidden_size"]])

        rnn = RNN(MyCell())
        outputs = []
        for i in range(n):
            input_i = rnn_input[:, i]
            input_i = input_i.transpose(...)
            output = rnn(input_i)
            outputs.append(output)
        output = keras.stack(outputs)

        # RNN
        rnn_cells = [GRUCell(hidden_size, activation="relu", dropout=dropout)
                     for hidden_size in self.config["en_rnn_hidden_sizes"]]
        # output dropout of GRUCell is omitted since the output_keep_prob is 1 in all configs.

    def _get_optional_configs(self):
        return {
            "batch_size",
            "dropout",
            "time_step",
            "filter_size",
            "long_num",
            "ar_size",
        }


