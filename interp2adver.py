import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import linear_model
import math, copy, sys, pickle


def GetCertainty(X, clf):
    proba = clf.predict_proba(X)
    return proba[:, 1] - proba[:, 0]


def GetAdversarySeeds(X, y, clf, ratio=0.04):     # ratio=0.03 for SVM, >=0.1 for LR and NN
    # Compute certainty
    certainty = GetCertainty(X, clf)          # signed distance to the boundary
    certainty[np.where(certainty<0)] = sys.maxint       # do not consider normal region

    # Choose instances based on certainty
    certainty_spammer = copy.copy(certainty)
    certainty_spammer[np.where(y==0)] = sys.maxint
    rank_certainty = np.argsort(certainty_spammer)
    num_adv_insts = int(X.shape[0]*ratio)
    adv_insts = X[rank_certainty[0:num_adv_insts],:]

    return adv_insts


def LocalInterpret_linear(xs, X_neg, clf, alpha_lasso=0.001, sampling_size=600):
    num_pos = xs.shape[0]
    dim = xs.shape[1]

    nbrs = NearestNeighbors(n_neighbors=5, p=2).fit(X_neg)

    # Estimate distortion distance for each instance
    dist2neg_xs = []
    for n in range(num_pos):
        x = copy.copy(xs[n, :])
        [dists_k, inds_k] = nbrs.kneighbors([x], 5, return_distance=True)
        dist2neg_xs.append(dists_k[0][0])
    avg_dis2neg = np.mean(dist2neg_xs)
    print "avg_dist2neg in original space: ", avg_dis2neg

    # Sampling and local interpretation using LASSO
    interpret_vecs = np.zeros((num_pos,dim))
    print "Number of adversarial samples: ", num_pos
    for n in range(num_pos):
        xs_samples = np.random.multivariate_normal(xs[n, :], dist2neg_xs[n]/math.sqrt(dim)/2.5*np.eye(dim), sampling_size)
        xs_samples = xs_samples.clip(min=0)
        labels_samples = clf.predict(xs_samples)
        lasso_itpr = linear_model.Lasso(alpha=alpha_lasso).fit(xs_samples, labels_samples)
        interpret_vecs[n,:] = copy.copy(lasso_itpr.coef_)

    return interpret_vecs, avg_dis2neg


def MoveInstances_l2(adv_insts, interpret_vecs, avg_dis2neg):
    num_insts = adv_insts.shape[0]
    adv_insts_new = copy.copy(adv_insts)
    for n in range(num_insts):
        len_move = np.linalg.norm(interpret_vecs[n, :])
        if len_move > 0:
            adv_insts_new[n, :] = adv_insts[n, :] - interpret_vecs[n, :] * avg_dis2neg / len_move
        else:
            adv_insts_new[n, :] = copy.copy(adv_insts[n, :])

    return adv_insts_new


def GenerateAdvSamplesLASSOL2(X_train, y_train, clf_raw, dist_ratio_arr, seed_ratio):
    # Evasion-prone samples selection
    adv_insts = GetAdversarySeeds(X_train, y_train, clf_raw, seed_ratio)

    # Local interpretation
    [interpret_vecs, avg_dis2neg] = LocalInterpret_linear(adv_insts, X_train[np.where(y_train==0)], clf_raw)
    print np.mean(interpret_vecs, axis=0)

    # Perturb seed instances according to the interpretation vec
    adv_insts_new = MoveInstances_l2(adv_insts, interpret_vecs, avg_dis2neg)
    adv_insts_batch = np.zeros([len(dist_ratio_arr), adv_insts_new.shape[0], adv_insts_new.shape[1]])
    for i in range(adv_insts_new.shape[0]):
        for d in range(len(dist_ratio_arr)):
            len_tmp = max(np.linalg.norm(adv_insts_new[i]-adv_insts[i]), 1e-9)
            adv_insts_batch[d, i] = adv_insts[i] + \
                               (adv_insts_new[i]-adv_insts[i]) / len_tmp * avg_dis2neg * dist_ratio_arr[d]

    np.save('adver.npy', adv_insts_batch)
    pickle.dump(clf_raw, open('model_old.sav', 'wb'))

    return adv_insts_batch
