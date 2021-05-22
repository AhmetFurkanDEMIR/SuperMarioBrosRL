# @author: Ahmet Furkan DEMIR

from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

class TFQModel(DistributionalQTFModel):
    """Custom model for DQN."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):


        super(TFQModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        self.layer_cnn_1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', strides=(2, 2))(self.inputs)
        self.layer_cnn_2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(self.layer_cnn_1)
        self.layer_cnn_3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', strides=(2, 2))(self.layer_cnn_2)
        self.layer_cnn_4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(self.layer_cnn_3)

        self.layer_lstm2D = tf.keras.layers.ConvLSTM2D(32, kernel_size=3, activation='relu')(tf.expand_dims(self.layer_cnn_4, axis=1))

        self.layer_flatten = tf.keras.layers.Flatten()(self.layer_lstm2D)
        self.layer_dense1 = tf.keras.layers.Dense(1024, activation='relu')(self.layer_flatten)
        self.layer_dense2 = tf.keras.layers.Dense(num_outputs, activation='tanh')(self.layer_dense1)
        self.base_model = tf.keras.Model(self.inputs, self.layer_dense2)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict["obs"])
        return model_out, state

    def metrics(self):
        return {"foo": tf.constant(42.0)}
