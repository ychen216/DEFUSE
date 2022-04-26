import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow import feature_column
from tensorflow.keras import regularizers

num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)


class MLP(Model):

    def __init__(self, name, params):
        super(MLP, self).__init__()
        self.model_name = name
        self.params = params
        num_features = [feature_column.bucketized_column(
            feature_column.numeric_column(str(i)),
            boundaries=[j/(num_bin_size[i]-1) for j in range(num_bin_size[i]-1)])
            for i in range(8)]
        cate_features = [feature_column.embedding_column(
            feature_column.categorical_column_with_hash_bucket(
                str(i), hash_bucket_size=cate_bin_size[i-8]),
            dimension=8) for i in range(8, 17)]

        all_features = num_features + cate_features

        self.feature_layer = tf.keras.layers.DenseFeatures(all_features)

        self.fc1 = layers.Dense(256, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
        self.bn1 = layers.BatchNormalization()
        self.fc2 = layers.Dense(256, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
        self.bn2 = layers.BatchNormalization()
        self.fc3 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))
        self.bn3 = layers.BatchNormalization()
        print("build model {}".format(name))
        if self.model_name in ["MLP_SIG", "MLP_dp"]:
            self.fc4 = layers.Dense(1)
        elif self.model_name in ["MLP_EXP_DELAY", "MLP_tn_dp", "Bi-DEFUSE_MLP"]:
            self.fc4 = layers.Dense(2)
        elif self.model_name in ["Bi-DEFUSE_inoutw"]:
            self.fc31 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                    kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn31 = layers.BatchNormalization()
            self.fc32 = layers.Dense(128, activation=tf.nn.leaky_relu,
                                    kernel_regularizer=regularizers.l2(params["l2_reg"]))
            self.bn32 = layers.BatchNormalization()
            self.attention0 = tf.compat.v1.get_variable('attention0', shape=[1, 2, 1], dtype=tf.float32)
                # initializer=tf.constant_initializer(1.0))
            self.attention1 = tf.compat.v1.get_variable('attention1', shape=[1, 2, 1], dtype=tf.float32)
                # initializer=tf.constant_initializer(1.0))
            self.fc41 = layers.Dense(1)
            self.fc42 = layers.Dense(1)
        else:
            raise ValueError("model name {} not exist".format(name))
        self.fc5 = layers.Dense(2)
        self.fc6 = layers.Dense(1, activation=tf.nn.leaky_relu,
                                kernel_regularizer=regularizers.l2(params["l2_reg"]))

    def call(self, x, training=True):
        x = self.feature_layer(x)
        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        if self.model_name == "Bi-DEFUSE_inoutw":
            x1 = self.fc31(x)
            x1 = self.bn31(x1) # inw logits
            x2 = self.fc31(x)
            x2 = self.bn31(x2) # outw logits
        x = self.fc3(x)
        x = self.bn3(x, training=training)
        # x_fix = self.fc5(x)
        if self.model_name == "MLP_EXP_DELAY":
            x = self.fc4(x)
            return {"logits": tf.reshape(x[:, 0], (-1, 1)), "log_lamb": tf.reshape(x[:, 1], (-1, 1))}
        elif self.model_name in ["MLP_SIG", "MLP_dp"]:
            x = self.fc4(x)
            return {"logits": x}
        elif self.model_name == "MLP_cvr_dp":
            x = self.fc4(x)
            return {"logits": tf.reshape(x[:, 0], (-1, 1)), "dp_logits": tf.reshape(x[:, 1], (-1, 1))}
        elif self.model_name == "Bi-DEFUSE_MLP":
            x = self.fc4(x)
            ret = {"logits_inw": tf.reshape(x[:, 0], (-1, 1)), "logits_outw": tf.reshape(x[:, 1], (-1, 1))}
            return ret
        elif self.model_name == "Bi-DEFUSE_inoutw":
            expert_tensor0 = tf.concat([x1, x], axis=1)
            expert_tensor1 = tf.concat([x2, x], axis=1)
            # num_hidden_units = num_hidden_units from previous layer(=last)
            expert_tensor0 = tf.reshape(expert_tensor0, [-1, 2, 128])
            expert_tensor1 = tf.reshape(expert_tensor1, [-1, 2, 128])
            expert_att0 = tf.reduce_sum(expert_tensor0 * self.attention0, axis=1)
            expert_att1 = tf.reduce_sum(expert_tensor1 * self.attention1, axis=1)
            x1 = self.fc41(expert_att0)
            x2 = self.fc42(expert_att1)
            ret = {"logits_inw": tf.reshape(x1[:, 0], (-1, 1)), "logits_outw": tf.reshape(x2[:, 0], (-1, 1))}
            return ret
        elif self.model_name == "MLP_tn_dp":
            x = self.fc4(x)
            return {"tn_logits": tf.reshape(x[:, 0], (-1, 1)), "dp_logits": tf.reshape(x[:, 1], (-1, 1))}
        else:
            raise NotImplementedError()

    def predict(self, x):
        if self.model_name in ["Bi-DEFUSE_MLP", "Bi-DEFUSE_inoutw"]:
            outs = self.call(x, training=False)
            inw_pred = tf.sigmoid(outs["logits_inw"])
            outw_pred = tf.sigmoid(outs["logits_outw"])
            # outs["fusion"] = self.fc6(tf.stop_gradient(outs["x"]))
            pred = inw_pred + outw_pred
            # return inw_pred + outw_pred, inw_pred, outw_pred
            return pred, inw_pred, outw_pred
            # return tf.sigmoid(outs["logits_inw"] + outs["logits_outw"])
            # return tf.sigmoid(outs["logits_inw"])
            # return tf.sigmoid(outs["logits_outw"])
        else:
            return self.call(x, training=False)["logits"]


def get_model(name, params):
    if name in ["MLP_EXP_DELAY", "MLP_SIG", "MLP_tn_dp", "MLP_dp", "Bi-DEFUSE_MLP", "Bi-DEFUSE_inoutw"]:
        return MLP(name, params)
    else:
        raise NotImplementedError()
