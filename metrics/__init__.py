import numpy as np

from .performance import compute_model_metrics
from .inference import get_frames_per_second, get_flops

def net_score(acc_score, n_parameters, n_flops, alpha=2, beta=0.5, gamma=0.5):
    ''' Compute NetScore '''
    score = 20 * np.log(
        np.power(acc_score, alpha) / (
            np.power(n_parameters, beta) * np.power(n_flops, gamma)
        )
    )
    return score
