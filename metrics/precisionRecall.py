import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_precision_recall(real_feats, fake_feats, k=3):
    real = real_feats.numpy() if hasattr(real_feats, 'numpy') else real_feats
    fake = fake_feats.numpy() if hasattr(fake_feats, 'numpy') else fake_feats
    nn_real = NearestNeighbors(n_neighbors=k).fit(real)
    real_dists, _ = nn_real.kneighbors(real)
    real_radius = real_dists.max(axis=1).mean()
    fake_dists, _ = nn_real.kneighbors(fake)
    precision = np.mean(fake_dists[:, -1] <= real_radius)
    nn_fake = NearestNeighbors(n_neighbors=k).fit(fake)
    fake_dists2, _ = nn_fake.kneighbors(fake)
    fake_radius = fake_dists2.max(axis=1).mean()
    real_dists2, _ = nn_fake.kneighbors(real)
    recall = np.mean(real_dists2[:, -1] <= fake_radius)
    return precision, recall