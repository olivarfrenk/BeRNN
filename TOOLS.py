import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import errno
import six
import json
import random
import pickle
import numpy as np

# todo: Functions similiar to Yang et al.'s Task.py ####################################################################
# todo: ################################################################################################################
rules_dict = {'all' : ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2',
              'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']}

rule_name = {
            'DM': 'Decison Making (DM)',
            'DM_Anti': 'Decision Making Anti (DM Anti)',
            'EF': 'Executive Function (EF)',
            'EF_Anti': 'Executive Function Anti (EF Anti)',
            'RP': 'Relational Processing (RP)',
            'RP_Anti': 'Relational Processing Anti (RP Anti)',
            'RP_Ctx1': 'Relational Processing Context 1 (RP Ctx1)',
            'RP_Ctx2': 'Relational Processing Context 2 (RP Ctx2)',
            'WM': 'Working Memory (WM)',
            'WM_Anti': 'Working Memory Anti (WM Anti)',
            'WM_Ctx1': 'Working Memory Context 1 (WM Ctx1)',
            'WM_Ctx2': 'Working Memory Context 2 (WM Ctx2)'
            }

# Store indices of rules
rule_index_map = dict()
for ruleset, rules in rules_dict.items():
    rule_index_map[ruleset] = dict()
    for ind, rule in enumerate(rules):
        rule_index_map[ruleset][rule] = ind

def get_num_ring(ruleset):
    '''get number of stimulus rings'''
    return 2 if ruleset=='all' else 2 # leave it felxible for potential future rulesets

def get_num_rule(ruleset):
    '''get number of rules'''
    return len(rules_dict[ruleset])

def get_rule_index(rule, config):
    '''get the input index for the given rule'''
    return rule_index_map[config['ruleset']][rule]+config['rule_start']

def get_dist(original_dist):
    '''Get the distance in periodic boundary conditions'''
    return np.minimum(abs(original_dist),2*np.pi-abs(original_dist))

def load_trials(trial_dir,monthsConsidered,task,mode):
    '''Load trials from pickle file'''
    # Build-in mechanism to prevent interruption of code as for many .npy files there errors are raised
    max_attempts = 30
    attempt = 0

    while attempt < max_attempts:
        # random choose one of the preprocessed files according to the current chosen task
        file_splits = random.choice(os.listdir(os.path.join(trial_dir,task))).split('-')
        while file_splits[1].split('_')[1] not in monthsConsidered:
            # randomly choose another file until the one for the right considered month is found
            file_splits = random.choice(os.listdir(os.path.join(trial_dir, task))).split('-')
        file_stem = file_splits[0]+'-'+file_splits[1]+'-'+file_splits[2]+'-'+file_splits[3]+'-'+file_splits[4]
        try:
            x = np.load(os.path.join(trial_dir, task, file_stem) + '-Input.npy', mmap_mode='r')
            y = np.load(os.path.join(trial_dir, task, file_stem) + '-Output.npy', mmap_mode='r')
            y_loc = np.load(os.path.join(trial_dir, task, file_stem) + '-yLoc.npy', mmap_mode='r')
            # Select rows for either training or evaluation
            if mode == 'Training': # get the first 32 rows
                x = x[:, :32, :]
                y = y[:, :32, :]
                y_loc = y_loc[:, :32]
            elif mode == 'Evaluation': # get the last 8 rows
                x = x[:, -8:, :]
                y = y[:, -8:, :]
                y_loc = y_loc[:, -8:]
            # print(file_stem, ' successfully processed.')
            return x,y,y_loc
        except Exception as e:
            print(f"An error occurred with file {file_stem}: {e}. Retrying...")
            attempt += 1
    if attempt == max_attempts:
        print("Maximum attempts reached. The function failed to execute successfully.")

# todo: ################################################################################################################
# todo: ################################################################################################################

def gen_feed_dict(model, x, y, hp):
    """Generate feed_dict for session run."""
    if hp['in_type'] == 'normal':
        feed_dict = {model.x: x,
                     model.y: y}
    elif hp['in_type'] == 'multi':
        n_time, batch_size = x.shape[:2]
        new_shape = [n_time,
                     batch_size,
                     hp['rule_start']*hp['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(x[0, i, hp['rule_start']:])
            i_start = ind_rule*hp['rule_start']
            x[:, i, i_start:i_start+hp['rule_start']] = \
                x[:, i, :hp['rule_start']]

        feed_dict = {model.x: x,
                     model.y: y}
    else:
        raise ValueError()

    return feed_dict

def _contain_model_file(model_dir):
    """Check if the directory contains model files."""
    for f in os.listdir(model_dir):
        if 'model.ckpt' in f:
            return True
    return False

def _valid_model_dirs(root_dir):
    """Get valid model directories given a root directory."""
    return [x[0] for x in os.walk(root_dir) if _contain_model_file(x[0])]

def valid_model_dirs(root_dir):
    """Get valid model directories given a root directory(s).

    Args:
        root_dir: str or list of strings
    """
    if isinstance(root_dir, six.string_types):
        return _valid_model_dirs(root_dir)
    else:
        model_dirs = list()
        for d in root_dir:
            model_dirs.extend(_valid_model_dirs(d))
        return model_dirs

def load_log(model_dir):
    """Load the log file of model save_name"""
    fname = os.path.join(model_dir, 'log.json')
    if not os.path.isfile(fname):
        return None

    with open(fname, 'r') as f:
        log = json.load(f)
    return log

def save_log(log):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, 'log.json')
    with open(fname, 'w') as f:
        json.dump(log, f)

def load_hp(model_dir):
    """Load the hyper-parameter file of model save_name"""
    fname = os.path.join(model_dir, 'hp.json')
    if not os.path.isfile(fname):
        fname = os.path.join(model_dir, 'hparams.json')  # backward compat
        if not os.path.isfile(fname):
            return None

    with open(fname, 'r') as f:
        hp = json.load(f)

    # Use a different seed aftering loading,
    # since loading is typically for analysis
    hp['rng'] = np.random.RandomState(hp['seed']+1000)
    return hp

def save_hp(hp, model_dir):
    """Save the hyper-parameter file of model save_name"""
    hp_copy = hp.copy()
    hp_copy.pop('rng')  # rng can not be serialized
    with open(os.path.join(model_dir, 'hp.json'), 'w') as f:
        json.dump(hp_copy, f)

def load_pickle(file):
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    return data

def find_all_models(root_dir, hp_target):
    """Find all models that satisfy hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters

    Returns:
        model_dirs: list of model directories
    """
    dirs = valid_model_dirs(root_dir)

    model_dirs = list()
    for d in dirs:
        hp = load_hp(d)
        if all(hp[key] == val for key, val in hp_target.items()):
            model_dirs.append(d)

    return model_dirs

def find_model(root_dir, hp_target, perf_min=None):
    """Find one model that satisfies hyperparameters.

    Args:
        root_dir: root directory
        hp_target: dictionary of hyperparameters
        perf_min: float or None. If not None, minimum performance to be chosen

    Returns:
        d: model directory
    """
    model_dirs = find_all_models(root_dir, hp_target)
    if perf_min is not None:
        model_dirs = select_by_perf(model_dirs, perf_min)

    if not model_dirs:
        # If list empty
        print('Model not found')
        return None, None

    d = model_dirs[0]
    hp = load_hp(d)

    log = load_log(d)
    # check if performance exceeds target
    if log['perf_min'][-1] < hp['target_perf']:
        print("""Warning: this network perform {:0.2f}, not reaching target
              performance {:0.2f}.""".format(
              log['perf_min'][-1], hp['target_perf']))

    return d

def select_by_perf(model_dirs, perf_min):
    """Select a list of models by a performance threshold."""
    new_model_dirs = list()
    for model_dir in model_dirs:
        log = load_log(model_dir)
        # check if performance exceeds target
        if log['perf_min'][-1] > perf_min:
            new_model_dirs.append(model_dir)
    return new_model_dirs

def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def gen_ortho_matrix(dim, rng=None):
    """Generate random orthogonal matrix
    Taken from scipy.stats.ortho_group
    Copied here from compatibilty with older versions of scipy
    """
    H = np.eye(dim)
    for n in range(1, dim):
        if rng is None:
            x = np.random.normal(size=(dim-n+1,))
        else:
            x = rng.normal(size=(dim-n+1,))
        # random sign, 50/50, but chosen carefully to avoid roundoff error
        D = np.sign(x[0])
        x[0] += D*np.sqrt((x*x).sum())
        # Householder transformation
        Hx = -D*(np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hx
        H = np.dot(H, mat)
    return H



###### DEBUG ###########################################################################################################


