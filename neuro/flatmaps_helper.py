from copy import deepcopy
from os.path import join

import joblib
import numpy as np

import neuro.analyze_helper
from neuro import config

FLATMAPS_DIR = join(config.FMRI_DIR_BLOB, 'brain_tune', 'flatmaps_all')
'''Note: these flatmaps were saved by the 03_resp_flatmaps.py script under GCT / 2_analyze
'''


def load_flatmaps(normalize_flatmaps, load_timecourse=False, explanations_only=False):
    # S02
    gemv_flatmaps_default = joblib.load(join(
        FLATMAPS_DIR, "UTS02", "default_pilot", 'resps_avg_dict_pilot.pkl'))
    gemv_flatmaps_qa = joblib.load(join(
        FLATMAPS_DIR, "UTS02", 'qa_pilot5', 'resps_avg_dict_pilot5.pkl'))
    gemv_flatmaps_roi = joblib.load(join(
        FLATMAPS_DIR, "UTS02", 'roi_pilot5', 'resps_avg_dict_pilot5.pkl'))

    gemv_flatmaps_roi_custom = joblib.load(join(
        FLATMAPS_DIR, 'UTS02', 'roi_pilot6', 'resps_avg_dict_pilot6.pkl'))
    gemv_flatmaps_dict_S02 = gemv_flatmaps_default | gemv_flatmaps_qa | gemv_flatmaps_roi | gemv_flatmaps_roi_custom
    # gemv_flatmaps_dict_S02 = gemv_flatmaps_roi_custom

    # S03
    gemv_flatmaps_default = joblib.load(join(
        FLATMAPS_DIR, 'UTS03', 'default', 'resps_avg_dict_pilot3.pkl'))
    gemv_flatmaps_roi_custom1 = joblib.load(join(
        FLATMAPS_DIR, 'UTS03', 'roi_pilot7', 'resps_avg_dict_pilot7.pkl'))
    gemv_flatmaps_roi_custom2 = joblib.load(join(
        FLATMAPS_DIR, 'UTS03', 'roi_pilot8', 'resps_avg_dict_pilot8.pkl'))
    # gemv_flatmaps_dict_S03 = gemv_flatmaps_default | gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2
    gemv_flatmaps_dict_S03 = gemv_flatmaps_default | gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2

    if load_timecourse:
        gemv_flatmaps_default = joblib.load(join(
            FLATMAPS_DIR, "UTS02", "default_pilot", 'resps_concat_dict_pilot.pkl'))
        gemv_flatmaps_qa = joblib.load(join(
            FLATMAPS_DIR, "UTS02", 'qa_pilot5', 'resps_concat_dict_pilot5.pkl'))
        gemv_flatmaps_roi = joblib.load(join(
            FLATMAPS_DIR, "UTS02", 'roi_pilot5', 'resps_concat_dict_pilot5.pkl'))
        gemv_flatmaps_roi_custom = joblib.load(join(
            FLATMAPS_DIR, 'UTS02', 'roi_pilot6', 'resps_concat_dict_pilot6.pkl'))

        gemv_flatmaps_dict_S02_timecourse = gemv_flatmaps_default | gemv_flatmaps_qa | gemv_flatmaps_roi | gemv_flatmaps_roi_custom
        # gemv_flatmaps_dict_S02_timecourse = gemv_flatmaps_roi_custom

        gemv_flatmaps_roi_custom1 = joblib.load(join(
            FLATMAPS_DIR, 'UTS03', 'roi_pilot7', 'resps_concat_dict_pilot7.pkl'))
        gemv_flatmaps_roi_custom2 = joblib.load(join(
            FLATMAPS_DIR, 'UTS03', 'roi_pilot8', 'resps_concat_dict_pilot8.pkl'))
        gemv_flatmaps_dict_S03_timecourse = gemv_flatmaps_roi_custom1 | gemv_flatmaps_roi_custom2

        return gemv_flatmaps_dict_S02, gemv_flatmaps_dict_S03, gemv_flatmaps_dict_S02_timecourse, gemv_flatmaps_dict_S03_timecourse

    # normalize flatmaps
    if normalize_flatmaps:
        for k, v in gemv_flatmaps_dict_S03.items():
            flatmap_unnormalized = gemv_flatmaps_dict_S03[k]
            gemv_flatmaps_dict_S03[k] = (
                flatmap_unnormalized - flatmap_unnormalized.mean()) / flatmap_unnormalized.std()
        for k, v in gemv_flatmaps_dict_S02.items():
            flatmap_unnormalized = gemv_flatmaps_dict_S02[k]
            gemv_flatmaps_dict_S02[k] = (
                flatmap_unnormalized - flatmap_unnormalized.mean()) / flatmap_unnormalized.std()

    return gemv_flatmaps_dict_S02, gemv_flatmaps_dict_S03


def get_weights_top(args, avg_over_delays=True):
    '''Return weights without delays out_size x
    '''

    model_params = joblib.load(
        join(args.save_dir_unique, 'model_params.pkl'))
    # print(f'{args.feature_space=}, {args.pc_components=}, {args.ndelays=} {args.qa_embedding_model}')

    # get weights
    ndelays = args.ndelays
    weights = model_params['weights']
    assert weights.shape[0] % ndelays == 0
    emb_size = weights.shape[0] / ndelays
    weights = weights.reshape(ndelays, int(emb_size), -1)
    if avg_over_delays:
        weights = weights.mean(axis=0)

    if hasattr(model_params, 'weights_pc'):
        weights_pc = model_params['weights_pc']
        assert weights_pc.shape[0] % ndelays == 0
        qs_size = weights_pc.shape[0] / ndelays
        weights_pc = weights_pc.reshape(ndelays, int(qs_size), -1)
        if avg_over_delays:
            weights_pc = weights_pc.mean(axis=0)
    else:
        weights_pc = None

    return weights, weights_pc


def load_custom_rois(subject, suffix_setting='_fedorenko'):
    '''
    Params
    ------
    subject: str
        'S02' or 'S03'
    suffix_setting: str
        '' - load custom communication rois
        '_fedorenko' - load fedorenko rois
        '_spotlights' - load spotlights rois (there are a ton of these)
    '''
    if suffix_setting == '':
        # rois_dict = joblib.load(join(regions_idxs_dir, f'rois_{subject}.jbl'))
        # rois = joblib.load(join(FMRI_DIR, 'brain_tune/voxel_neighbors_and_pcs/', 'communication_rois_UTS02.jbl'))
        rois = joblib.load(join(config.FMRI_DIR_BLOB, 'brain_tune/voxel_neighbors_and_pcs/',
                                f'communication_rois_v2_UT{subject}.jbl'))
        rois_dict_raw = {i: rois[i] for i in range(len(rois))}
        if subject == 'S02':
            raw_idxs = [
                [0, 7],
                [3, 4],
                [1, 5],
                [2, 6],
            ]
        elif subject == 'S03':
            raw_idxs = [
                [0, 7],
                [3, 4],
                [2, 5],
                [1, 6],
            ]
        return {
            'comm' + str(i): np.vstack([rois_dict_raw[j] for j in idxs]).sum(axis=0)
            for i, idxs in enumerate(raw_idxs)
        }
    elif suffix_setting == '_fedorenko':
        if subject == 'S03':
            rois_fedorenko = joblib.load(join(
                config.FMRI_DIR_BLOB, 'brain_tune/voxel_neighbors_and_pcs/', 'lang_localizer_UTS03.jbl'))
        elif subject == 'S02':
            rois_fedorenko = joblib.load(join(
                config.FMRI_DIR_BLOB, 'brain_tune/voxel_neighbors_and_pcs/', 'lang_localizer_UTS02_aligned.jbl'))
        return {
            'Lang-' + str(i): rois_fedorenko[i] for i in range(len(rois_fedorenko))
        }
        # rois_dict = rois_dict_raw
    elif suffix_setting == '_spotlights':
        rois_spotlights = joblib.load(join(
            config.FMRI_DIR_BLOB, 'brain_tune/voxel_neighbors_and_pcs/', f'all_spotlights_UT{subject}.jbl'))
        return {'spot' + str(i): rois_spotlights[i][-1]
                for i in range(len(rois_spotlights))}
    elif suffix_setting == '_lobes':
        lobes_by_subj = np.load(
            join(config.FMRI_DIR_BLOB, 'brain_tune', 'lobes_vox.npz'), allow_pickle=True)
        lobes_dict_by_hemisphere = lobes_by_subj[subject].flatten()[0]
        lobes_dict = {}
        for k, v in lobes_dict_by_hemisphere.items():
            lobes_dict[k] = np.zeros(neuro.analyze_helper.VOX_COUNTS[subject])
            v_left, v_right = v
            lobes_dict[k][v_left] = 1
            lobes_dict[k][v_right] = 1
        # cast values as type int instead of bool
        # lobes_dict = {k: v.astype(int) for k, v in lobes_dict.items()}
        return lobes_dict


ROI_EXPLANATIONS_S03 = {
    'EBA': 'Body parts',
    'IPS': 'Descriptive elements of scenes or objects',
    'OFA': 'Conversational transitions',
    'OPA': 'Direction and location descriptions',
    'OPA_only': 'Self-reflection and growth',
    'PPA': 'Scenes and settings',
    'PPA_only': 'Garbage, food, and household items',
    'RSC': 'Travel and location names',
    'RSC_only': 'Location names',
    'sPMv': 'Dialogue and responses',
}

FED_DRIVING_EXPLANATIONS_S03 = {
    0: 'Relationships',
    1: 'Positive Emotional Reactions',
    2: 'Body parts',
    3: 'Dialogue',
    4: 'Negative Emotional Reactions',
}

FED_DRIVING_EXPLANATIONS_S02 = {
    0: 'Secretive Or Covert Actions',
    1: 'Introspection',
    2: 'Relationships',
    3: 'Sexual and Romantic Interactions',
    4: 'Dialogue',
}


def load_known_rois(subject):
    nonzero_entries_dict = joblib.load(
        join(config.REGION_IDXS_DIR, f'rois_{subject}.jbl'))
    rois_dict = {}
    for k, v in nonzero_entries_dict.items():
        mask = np.zeros(neuro.analyze_helper.VOX_COUNTS[subject])
        mask[v] = 1
        rois_dict[k] = deepcopy(mask)
    if subject == 'S03':
        rois_dict['OPA'] = rois_dict['TOS']
    return rois_dict


if __name__ == '__main__':
    lobes_dict = load_custom_rois('S02', '_lobes')
    print(lobes_dict.keys())
