import os
import time
from config import DIC_AGENTS

class Pipeline:

    def _path_check(self):
        # check path
        if not os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])
        elif self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError

        if not os.path.exists(self.dic_path["PATH_TO_MODEL"]):
            os.makedirs(self.dic_path["PATH_TO_MODEL"])
        elif self.dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError

    def __init__(self, dic_exp_conf, dic_agent_conf, dic_path):

        # load configurations
        self.dic_exp_conf = dic_exp_conf
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path

    def updater_wrapper(self, cnt_round, dic_agent_conf, dic_path):

        agent_name = self.dic_agent_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name][0](dic_agent_conf, dic_path,cnt_round)
        agent.train_network()

    def run(self):

        cnt_round = self.dic_exp_conf["START_ROUNDS"]-1
        while cnt_round < self.dic_exp_conf["START_ROUNDS"]+self.dic_exp_conf["NUM_ROUNDS"]-1:
            cnt_round += 1

            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            if self.dic_exp_conf["MODEL_NAME"] in self.dic_exp_conf["LIST_MODEL_NEED_TO_UPDATE"]:
                print("==============  update network =============")
                self.updater_wrapper(cnt_round=cnt_round,
                                     dic_agent_conf=self.dic_agent_conf,
                                     dic_path=self.dic_path)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))

