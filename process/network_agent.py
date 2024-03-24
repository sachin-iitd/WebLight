
import numpy as np
from keras import losses
from keras.layers import Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping
import os
import traceback
import google

class NetworkAgent:
    def __init__(self, dic_agent_conf, dic_path, cnt_round):
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path
        self.num_out = dic_agent_conf["NUM_OUT"]
        self.cnt_round = cnt_round
        if cnt_round == 0:
            # initialization
            self.q_network = self.build_network()
        else:
            try:
                self.load_network("round_{0}".format(cnt_round - 1))
            except Exception as e:
                print('traceback.format_exc():\n%s' % traceback.format_exc())

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_sep_branch_{0}_1".format(memo))(state_features)
        q_values = Dense(num_actions, activation="linear", name="q_val_sep_branch_{0}".format(memo))(hidden_1)
        return q_values

    def load_network(self, file_name, file_path=None):
        if file_path == None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name))
        print("succeed in loading model %s"%file_name, 'at load_network')

    def save_network(self, file_name):
        self.q_network.save(os.path.join(self.dic_path["PATH_TO_MODEL"], "%s.h5" % file_name))

    def build_network(self):
        raise NotImplementedError

    def train_network(self):

        for cnt_round in range(self.cnt_round, self.dic_agent_conf['NUM_ROUNDS']):

            X, Y = google.load_samples(self.dic_agent_conf["BATCH_SIZE"])
            batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(Y))

            early_stopping = EarlyStopping(
                monitor='val_loss', patience=self.dic_agent_conf["PATIENCE"], verbose=0, mode='min')

            hist = self.q_network.fit(X, Y, batch_size=batch_size, epochs=self.dic_agent_conf["EPOCHS"],
                                      shuffle=False,
                                      verbose=0, validation_split=0.1, callbacks=[early_stopping])
            self.save_network("round_{0}".format(cnt_round))

            # Analyze
            testX, testY = google.load_validate()
            out = self.q_network.predict(testX)
            mse = losses.mean_squared_error(testY, out)
            mse = sum(mse)/len(mse)
            diff = np.abs(np.subtract(testY, out))
            percent = (diff/testY)*100
            res = "{0:.2f},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.2f},{6:.2f},{7:.2f},{8:.2f},{9:.2f},{10:.2f},{11:.2f},{12:.2f},{13:.2f}".format(
                hist.history['loss'][-1], hist.history['val_loss'][-1], mse, np.min(percent[:,0]), np.min(percent[:,1]), np.min(percent[:,2]), np.min(percent[:,3]), np.mean(percent[:,0]), np.mean(percent[:,1]), np.mean(percent[:,2]), np.mean(percent[:,3]), np.max(percent[:,1]), np.max(percent[:,2]), np.max(percent[:,3]))
            print(cnt_round, ': Metrics :', res.replace(',', '  '))
            file_name = self.dic_path['PATH_TO_WORK_DIRECTORY'] + '/metrics.csv'
            mode = 'a'
            if not os.path.exists(file_name):
                mode = 'w'
            with open(file_name, mode) as file:
                print(res, file=file)

