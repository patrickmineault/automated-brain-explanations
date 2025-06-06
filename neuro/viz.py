import os
import os.path

import cortex
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

MODELS_RENAME = {
    'bert-base-uncased': 'BERT (Finetuned)',
    'bert-10__ndel=4fmri': 'BERT+fMRI (Finetuned)',
}


def savefig(fname, *args, **kwargs):
    """
    Save figure to file
    """
    if not os.path.dirname(fname) == '':
        os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, *args, **kwargs)
    # plt.savefig(fname.replace(".pdf", ".png"),
    # transparent=True, bbox_inches='tight')
    plt.close()


def feature_space_rename(x):
    FEATURE_SPACE_RENAME = {
        'bert-base-uncased': 'BERT',
        'eng1000': 'Eng1000',
        'qa_embedder': 'QA-Emb',
        'llama': 'LLaMA',
        'finetune_roberta-base-10': 'QA-Emb (distill, probabilistic)',
        'finetune_roberta-base_binary-10': 'QA-Emb (distill, binary)',
    }
    if x in FEATURE_SPACE_RENAME:
        return FEATURE_SPACE_RENAME[x]
    x = str(x)
    x = x.replace('-10', '')
    x = x.replace('llama2-70B', 'LLaMA-2 (70B)')
    x = x.replace('llama2-7B', 'LLaMA-2 (7B)')
    x = x.replace('llama3-8B', 'LLaMA-3 (8B)')
    x = x.replace('mist-7B', 'Mistral (7B)')
    x = x.replace('ensemble1', 'Ensemble')
    if '_lay' in x:
        x = x.replace('_lay', ' (lay ') + ')'
        x = x.replace('(lay 6)', '(lay 06)')
    return x


def version_rename(x):
    if x == 'v1':
        return 'Prompts 1-3 (376 questions)'
    elif x == 'v2':
        return 'Prompts 1-5 (518 questions)'
    elif x == 'v3_boostexamples':
        return 'Prompts 1-6 (674 questions)'
    else:
        return x


DSETS_RENAME = {
    'tweet_eval': 'Tweet Eval',
    'sst2': 'SST2',
    'rotten_tomatoes': 'Rotten tomatoes',
    'moral_stories': 'Moral stories',
}


def dset_rename(x):
    if x in DSETS_RENAME:
        return DSETS_RENAME[x]
    else:
        x = x.replace('probing-', '')
        x = x.replace('_', ' ')
        return x.capitalize()


def quickshow(
        X: np.ndarray, subject="UTS03", fname_save=None, cmap='RdBu_r',
        with_colorbar=True, kwargs={'with_rois': True}, cmap_perc_to_hide=None, center=True):
    """
    Actual visualizations
    Note: for this to work, need to point the cortex config filestore to the `ds003020/derivative/pycortex-db` directory.
    This might look something like `/home/chansingh/mntv1/deep-fMRI/data/ds003020/derivative/pycortex-db/UTS03/anatomicals/`
    """
    if isinstance(cmap, int):
        cmap = sns.color_palette("husl", cmap, as_cmap=True)

    if isinstance(X, cortex.VolumeRGB):
        vol = X
    else:
        if isinstance(X, cortex.Volume):
            vol = X
            X = vol.data
        else:
            if subject == 'fsaverage':
                xfmname = 'atlas_2mm'
            else:
                if subject.startswith('S0'):
                    subject = 'UT' + subject
                xfmname = f"{subject}_auto"

            vol = cortex.Volume(X, subject, xfmname=xfmname, cmap=cmap)
        # , with_curvature=True, with_sulci=True)
        vabs = np.nanmax(np.abs(X))
        if center:
            if not cmap == 'Reds':
                vol.vmin = -vabs
                vol.vmax = vabs
            if (cmap == 'Reds' or not center):
                vol.vmin = np.nanmin(X)
                vol.vmax = np.nanmax(X)

        if cmap_perc_to_hide is not None:
            vol.vmin = np.nanpercentile(X, cmap_perc_to_hide)
            vol.vmax = np.nanpercentile(X, 100 - cmap_perc_to_hide)
    # fig = plt.figure()
    # , vmin=-vabs, vmax=vabs)
    cortex.quickshow(vol, with_colorbar=with_colorbar, **kwargs)
    # fig = plt.gcf()
    # add title
    # fig.axes[0].set_title(title, fontsize='xx-small')
    if fname_save is not None:
        if not os.path.dirname(fname_save) == '':
            os.makedirs(os.path.dirname(fname_save), exist_ok=True)
        plt.savefig(fname_save)
        plt.savefig(fname_save.replace(".pdf", ".png"),
                    transparent=True, bbox_inches='tight')
        plt.close()
