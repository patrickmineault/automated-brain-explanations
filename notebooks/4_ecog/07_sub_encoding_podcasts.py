import os
import sys
from os.path import dirname, join

from imodelsx import submit_utils

path_to_file = dirname(os.path.abspath(__file__))
repo_dir = dirname(dirname(path_to_file))
sys.path.append(repo_dir)

params_shared_dict = {
    'out_dir': [join(path_to_file, 'encoding_podcasts_results')],
    # 'model_name': ['en_core_web_lg', 'syntactic', 'phonetic', 'spectral', 'whisper-medium']
}
params_coupled_dict = {
    ('model_name', 'layer'):
    [
        ('gpt2-xl', 24),
        ('en_core_web_lg', -1),
        ('syntactic', -1),
        ('phonetic', -1),
        ('spectral', -1),
        # ('whisper-medium', -1)
    ]
}

args_list = submit_utils.get_args_list(
    params_shared_dict=params_shared_dict,
    params_coupled_dict=params_coupled_dict,
)

# args_list = args_list[:1]

script_name = join(path_to_file, '07_encoding_podcasts.py')
amlt_kwargs = {
    # change this to run a cpu job
    'amlt_file': join(repo_dir, 'scripts', 'launch.yaml'),
    # [64G16-MI200-IB-xGMI, 64G16-MI200-xGMI
    'sku': '64G8-MI200-xGMI',
    # 'sku': '64G4-MI200-xGMI',
    # 'sku': '64G2-MI200-xGMI',
    'mnt_rename': ('/home/chansingh/mntv1', '/mntv1'),
}
# print(args_list)
submit_utils.run_args_list(
    args_list,
    script_name=script_name,
    # unique_seeds='seed_stories',
    # amlt_kwargs=amlt_kwargs,
    # gpu_ids=[0, 1],
    gpu_ids=[0, 1, 2, 3],
    # gpu_ids=[[0, 1], [2, 3]],
    # actually_run=False,
    # shuffle=True,
    # cmd_python=f'export HF_TOKEN={open(expanduser("~/.HF_TOKEN"), "r").read().strip()}; python',
    cmd_python='python',
)
