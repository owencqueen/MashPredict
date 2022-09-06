import os
import torch
import argparse
import numpy as np

base = '/lustre/isaac/scratch/oqueen/MashPredict/nn_models/cluster_runs'

def extract_summary(size):

    dir_path = os.path.join(base, 'scores', 's' + size)

    scores = []
    latscores = []
    lonscores = []

    for f in os.listdir(dir_path):
        if f[-3:] != '.pt':
            # Filter out any __pycache__ or .DS_Store
            continue

        overall, lat, lon = torch.load(os.path.join(dir_path, f))
        scores.append(overall)
        latscores.append(lat)
        lonscores.append(lon)

    return scores, latscores, lonscores


def main(size):

    scores, latscores, lonscores = extract_summary(size)

    def pscore(s):
        print('Score = {:.4f} +- {:.4f}'.format(np.mean(s), (np.std(s) / np.sqrt(len(s)))))

    print('Overall dist:', '-' * 30)
    pscore(scores)
    print('Lat dist:    ', '-' * 30)
    pscore(latscores)
    print('Long dist:   ', '-' * 30)
    pscore(lonscores)

if __name__ == '__main__':

    # Summarize results from one run of the mashnet
    parser = argparse.ArgumentParser()
    parser.add_argument('--mashsize', type = str, help = 'options: 500, 2k, 4k, 50k')

    args = parser.parse_args()

    assert args.mashsize in ['500', '2k', '4k', '50k'], "mashsize must be in ['500', '2k', '4k', '50k']"

    main(args.mashsize)
