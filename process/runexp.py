import config
import copy
from pipeline import Pipeline
import sys
import time
import argparse
import os
import random
import google

PRETRAIN=False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ggl", type=str, default="GT823")
    parser.add_argument("--gstate", type=str, default='1,5,9,13,17,21')
    parser.add_argument("--gout", type=str, default='38,39,40,41')

    parser.add_argument("--msg", type=str, default=None)
    parser.add_argument("--dense", type=str, default=None)
    parser.add_argument("--num_phase", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=1000)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dir_suffix", type=str, default=None)

    parser.add_argument("--memo", type=str, default='test')
    parser.add_argument("--mod", type=str, default="SimpleDQN")
    parser.add_argument("-dryrun", action="store_true", default=False)
    parser.add_argument("--visible_gpu", type=str, default="")

    return parser.parse_args()

def memo_rename(traffic_file_list):
    new_name = ""
    for traffic_file in traffic_file_list:
        if "synthetic" in traffic_file:
            sta = traffic_file.rfind("-") + 1
            print(traffic_file, int(traffic_file[sta:-4]))
            new_name = new_name + "syn" + traffic_file[sta:-4] + "_"
        elif "cross" in traffic_file:
            sta = traffic_file.find("equal_") + len("equal_")
            end = traffic_file.find(".xml")
            new_name = new_name + "uniform" + traffic_file[sta:end] + "_"
        elif "flow" in traffic_file:
            new_name = traffic_file[:-4]
    new_name = new_name[:-1]
    return new_name

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def check_all_workers_working(list_cur_p):
    for i in range(len(list_cur_p)):
        if not list_cur_p[i].is_alive():
            return i
    return -1

def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf,
                   dic_agent_conf=dic_agent_conf,
                   dic_path=dic_path
                   )
    ppl.run()
    print("pipeline_wrapper end")
    return


def main(args = None):

    traffic_file = args.ggl
    num_rounds = args.num_rounds
    num_intersections = 1
    print('num_intersections:', num_intersections)

    deploy_dic_exp_conf = {
        "NUM_ROUNDS": 1,    # For outer loop
        "START_ROUNDS": 0,
        "MODEL_NAME": args.mod,
        "LIST_MODEL_NEED_TO_UPDATE": ["SimpleDQN"]
    }

    dic_agent_conf_extra = {
        "EPOCHS": 100,
        "SAMPLE_SIZE": 1000,
        "MODEL_NAME": args.mod,
        "NUM_ROUNDS": num_rounds,   # For inner loop
        "N_LAYER": args.num_layers,
        "LIST_STATE_FEATURE": [],
        "DIC_FEATURE_DIM": dict(),
    }

    if args.resume is None:
        suffix = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))+'_'+(args.dir_suffix if args.dir_suffix is not None else str(random.randint(1,100)))
    else:
        resume_list = args.resume.split(',')
        suffix = resume_list[0]
        deploy_dic_exp_conf["START_ROUNDS"] = int(resume_list[1])

    dic_path_extra = {
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, traffic_file + "_" + suffix),
        "PATH_TO_MODEL": os.path.join("records", args.memo, traffic_file + "_" + suffix, 'model'),
        "PATH_TO_DATA": os.path.join("data"),
        "PATH_TO_ERROR": os.path.join("errors", args.memo)
    }

    deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(args.mod.upper())),
                                  dic_agent_conf_extra)
    deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

    if args.dense is not None:
        deploy_dic_agent_conf["D_DENSE"] = list(int(i) for i in args.dense.split(','))
        if deploy_dic_agent_conf["N_LAYER"] < len(deploy_dic_agent_conf["D_DENSE"]):
            deploy_dic_agent_conf["N_LAYER"] = len(deploy_dic_agent_conf["D_DENSE"])
            print('Increasing N_LAYER to', deploy_dic_agent_conf["N_LAYER"])
        print('D_DENSE:', deploy_dic_agent_conf["D_DENSE"])

    print('Google Time correlation analysis for file', args.ggl)
    state = [int(i) for i in args.gstate.split(',')]
    out = [int(i) for i in args.gout.split(',')]
    deploy_dic_agent_conf["LIST_STATE_FEATURE"] = ['ggl']
    deploy_dic_agent_conf["DIC_FEATURE_DIM"]['D_GGL'] = (len(state),)
    deploy_dic_agent_conf["NUM_OUT"] = len(out)
    deploy_dic_agent_conf['BATCH_SIZE'] = 1000
    google.load_data('data/'  + args.ggl + '.csv', state, out)

    # Print State and Rewards
    print('ARGS', args)
    if args.dryrun:
        raise "Dry Run"

    # Link model to records
    if args.resume is None:
        model_dir = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "model")
        os.makedirs(model_dir)

    code_dir = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "code_"+time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
    os.makedirs(code_dir)
    os.system('cp code/*.py ' + code_dir)

    # Write the msg
    if args.msg is not None:
        with open(os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "readme.txt"), 'a') as file:
            print('\n', time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())), file=file)
            print(' '.join(sys.argv[:]), file=file)
            args.msg = ''
            print(args, file=file)

    pipeline_wrapper(dic_exp_conf=deploy_dic_exp_conf,
                     dic_agent_conf=deploy_dic_agent_conf,
                     dic_path=deploy_dic_path)

    # Mark End of Experiment
    os.system('echo '+deploy_dic_path['PATH_TO_WORK_DIRECTORY'] + ' >> ' + os.path.join("records", args.memo, 'ExpDone.txt'))

    return args.memo


if __name__ == "__main__":
    args = parse_args()
    main(args)
