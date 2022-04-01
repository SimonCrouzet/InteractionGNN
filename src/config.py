from src.model import GAT, GCN, LinearNetwork
from configparser import ConfigParser, Error
from datetime import datetime


def read_config(path):
    config_object = ConfigParser()
    config_object.read(path)

    
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hh%Mm%Ss")
    model_no = "run_" + str(dt_string)

    runinfo = config_object['RUNINFO']
    modelinfo = config_object['MODELINFO']
    runparams = config_object['RUNPARAMS']
    runinfo['model_no'] = model_no

    return runinfo, modelinfo, runparams

# functions to read config file
def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False

def str_to_model(s):
    if s == "GCN":
        return GCN
    elif s == "GAT":
        return GAT
    elif s == "Linear":
        return LinearNetwork
    else:
        raise Error("Model error: {} not recognized as a model".format(s))