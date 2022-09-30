import os
import torch
import argparse
import numpy as np
from geopy.distance import geodesic

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

def compute_distance_error(gt, pred):
    #print(gt.shape)
    mean_loss = np.mean(np.sqrt([geodesic(gt[i,:], pred[i,:]).km ** 2 for i in range(gt.shape[0])]))
    return mean_loss

def pscore(s):
    print('Score = {:.4f} +- {:.4f}'.format(np.mean(s), (np.std(s) / np.sqrt(len(s)))))

def main(size):

    scores, latscores, lonscores = extract_summary(size)

    print('Overall dist:', '-' * 30)
    pscore(scores)
    print('Lat dist:    ', '-' * 30)
    pscore(latscores)
    print('Long dist:   ', '-' * 30)
    pscore(lonscores)

def main_from_preds(size):
    # Gather all scores:
    d = os.path.join('preds', 's' + size)

    scores = []
    i = 0
    for f in os.listdir(d):
        i += 1
        if i > 30:
            break
        sc = torch.load(os.path.join(d, f))
        #print('len sc', len(sc))
        scores_i = []
        for s in sc:
            # print('len s', len(s))
            # print('len s[0]', s[0].shape)
            # print('len s[1]', s[1].shape)
            # print(s[0])
            # print(s[1])
            gt, pred = s[0].numpy().T, s[1].numpy().T
            scores_i.append(compute_distance_error(gt, pred))

        scores.append(np.mean(scores_i))
    
    pscore(scores)

if __name__ == '__main__':

    # Summarize results from one run of the mashnet
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_preds', action = 'store_true', help = 'Computes base from predictions')
    parser.add_argument('--mashsize', type = str, help = 'options: 500, 2k, 4k, 50k')

    args = parser.parse_args()

    assert args.mashsize in ['500', '2k', '4k', '50k'], "mashsize must be in ['500', '2k', '4k', '50k']"

    if args.from_preds:
        main_from_preds(args.mashsize)
    else:
        main(args.mashsize)
