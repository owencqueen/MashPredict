import torch 
import argparse, os

from simple_net import *

base = '/lustre/isaac/scratch/oqueen/MashPredict'
map_to_filepaths = {
    '500': os.path.join(base, 'data/onehot_s500.txt'),
    '2k': os.path.join(base, 'data/onehot_s2000.txt'),
    '4k': os.path.join(base, 'data/onehot_s4000.txt'),
    '50k': os.path.join(base, 'data/trimmed_50000.txt')
}

def get_random_state(i):
    # Deterministic method to compute random state for model
    return i * 587 % 103 + 127

def main(mashsize, cv):

    pred_gt_path = os.path.join(base, 
        'nn_models/cluster_runs/preds/s{}/cv={}.pt'.format(mashsize, cv))
    scores_path = os.path.join(base, 
        'nn_models/cluster_runs/scores/s{}/cv={}.pt'.format(mashsize, cv))

    rs = get_random_state(cv)

    pred_gt, scores, lat_score, lon_score = cross_validate_screen(
        standard_scale_ll = True,
        random_state = rs,
        data_path = map_to_filepaths[mashsize],
        mpath_base = 's{}_cv={}'.format(mashsize, cv),
        rm_mpath = True
    )

    torch.save(pred_gt, pred_gt_path)
    torch.save((scores, lat_score, lon_score), scores_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', type = int, required = True,
        help = 'Cross validation number, for storage reasons.')
    parser.add_argument('--mashsize', type = str, help = 'options: 500, 2k, 4k, 50k')
    parser.add_argument('--num_cv', type = int, default = 1)

    args = parser.parse_args()

    assert args.mashsize in ['500', '2k', '4k', '50k'], "mashsize must be in ['500', '2k', '4k', '50k']"

    init_cv = args.cv

    for i in range(args.num_cv):
        main(args.mashsize, init_cv + i)
    
