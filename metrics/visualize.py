import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(real_feats, fake_feats, save_path=None):
    X = np.vstack([real_feats, fake_feats])
    y = np.array([0]*len(real_feats) + [1]*len(fake_feats))
    tsne = TSNE(n_components=2)
    X_emb = tsne.fit_transform(X)
    plt.figure(figsize=(6,6))
    plt.scatter(X_emb[y==0,0], X_emb[y==0,1], label='real', alpha=0.6)
    plt.scatter(X_emb[y==1,0], X_emb[y==1,1], label='fake', alpha=0.6)
    plt.legend()
    plt.title('t-SNE of real vs. synthetic')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()