import yaml
from box import Box

def read_config_from_file(cfg_file):
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    yaml_cfg = Box(yaml_cfg)
    return yaml_cfg
