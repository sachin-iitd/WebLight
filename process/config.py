# parameters and paths
from dqn_agent import SimpleDQNAgent

DIC_SIMPLEDQN_AGENT_CONF = {
    "LEARNING_RATE": 0.001,
    "SAMPLE_SIZE": 1000,
    "BATCH_SIZE": 20,
    "EPOCHS": 100,
    "GAMMA": 0.8,
    "MAX_MEMORY_LEN": 10000,
    "PATIENCE": 10,
    "D_DENSE": [20],
    "N_LAYER": 2,
    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
    "NORMAL_FACTOR": 20,
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_PRETRAIN_WORK_DIRECTORY": "records/default",
    "PATH_TO_PRETRAIN_DATA": "data/template",
    "PATH_TO_AGGREGATE_SAMPLES": "records/initial",
    "PATH_TO_ERROR": "errors/default",
    "PATH_TO_TEACHER_MODEL": None
}

DIC_AGENTS = {
    "SimpleDQN": [SimpleDQNAgent],
}
