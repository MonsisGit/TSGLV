import h5py
import json
import torch
from tqdm import tqdm
from torch.functional import F
from utils_benchmark import _get_movies_durations, _create_mask, compute_proposals, _compute_proposals_feats, _nms, \
    _pretty_print_results, _iou
import numpy as np
from matplotlib import pyplot as plt
import json

# Load annotations
SPLIT = 'test'
root = '/nfs/data3/goldhofer/mad_dataset'
test_data = json.load(open(f'{root}/annotations/MAD_{SPLIT}.json', 'r'))
annotations_keys = list(test_data.keys())
movies_durations = _get_movies_durations(annotations_keys, test_data)

# Load features
FPS = 5
video_feats = h5py.File(f'{root}/CLIP_frames_features_5fps.h5', 'r')
lang_feats = h5py.File(f'{root}/CLIP_language_features_MAD_{SPLIT}.h5', 'r')
grid = {}

TOP_K_FRAMES =20

for WINDOW_LENGTH in [25, 50, 120]:
    try:

        nm_queries_run = 0
        inside_top10_proposals = 0
        inside_top50_proposals = 0
        inside_top20_proposals = 0

        num_frames = 64
        num_input_frames = WINDOW_LENGTH * FPS
        test_stride = int(WINDOW_LENGTH / 2 * FPS)
        MASK = _create_mask(num_frames, [5, 8, 8, 8])
        proposals = compute_proposals(num_frames, num_input_frames, test_stride, MASK, movies_durations, float(FPS))

        # Define metric parameters
        iou_metrics = torch.tensor([0.1, 0.3, 0.5])
        num_iou_metrics = len(iou_metrics)

        recall_metrics = torch.tensor([1, 5, 10, 50, 100])
        max_recall = recall_metrics.max()
        num_recall_metrics = len(recall_metrics)
        recall_x_iou = torch.zeros((num_recall_metrics, len(iou_metrics)))

        proposals_features = {}
        cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity_ranking = {}
        nm_proposals = []

        # Computer performance
        for k in tqdm(annotations_keys):
            movie = test_data[k]['movie']
            prop = proposals[movie]
            # windows_idx = torch.round(prop * FPS).int()
            # windows_idx = torch.arange(0, movies_durations[movie], 1)
            gt_grounding = torch.tensor(test_data[k]['ext_timestamps'])
            nm_queries_run += 1
            # Get movie features and sentence features
            l_feat = torch.tensor(lang_feats[k], dtype=torch.float)[None, :]

            try:
                p_feats = proposals_features[movie]
            except:
                v_feat = torch.tensor(video_feats[movie], dtype=torch.float)
                # p_feats = _compute_proposals_feats(v_feat, windows_idx)
                p_feats = v_feat
                proposals_features[movie] = p_feats

            #mean_pooled_v_feats = v_feat[prop[0].long()[0]:prop[1].long()[0],:].mean(axis=0)

            v_feat_shifted_forward = v_feat[int((WINDOW_LENGTH * FPS)/2):, :]
            v_feat_shifted_backwards = v_feat[:-int((WINDOW_LENGTH * FPS)), :]

            v_feat_shifted_forward = v_feat_shifted_forward[0:min(v_feat_shifted_backwards.shape[0], v_feat_shifted_forward.shape[0]), :]
            v_feat_shifted_backwards = v_feat_shifted_backwards[0:min(v_feat_shifted_backwards.shape[0], v_feat_shifted_forward.shape[0]), :]

            rm_ind = v_feat_shifted_forward.shape[0] - v_feat_shifted_forward.shape[0] % (WINDOW_LENGTH * FPS)
            v_feat_shifted_forward = v_feat_shifted_forward[0:rm_ind, :]
            v_feat_shifted_backwards = v_feat_shifted_backwards[0:rm_ind, :]

            v_feat_shifted_forward = torch.reshape(v_feat_shifted_forward,(WINDOW_LENGTH*FPS,512,-1)).mean(axis=0).reshape(-1,512)
            v_feat_shifted_backwards = torch.reshape(v_feat_shifted_backwards, (WINDOW_LENGTH * FPS, 512, -1)).mean(axis=0).reshape(-1, 512)

            v_feat_concat = torch.zeros((v_feat_shifted_forward.shape[0] * 2, v_feat_shifted_forward.shape[1]))
            v_feat_concat[::2] = v_feat_shifted_backwards
            v_feat_concat[1::2] = v_feat_shifted_forward

            sim = cosine_similarity(l_feat, v_feat_concat)
            # best_moments = _nms(prop, sim, topk=recall_metrics[-1], thresh=0.3)

            # mious = _iou(best_moments[:max_recall], gt_grounding)
            # bools = mious[:, None].expand(max_recall, num_iou_metrics) > iou_metrics
            # for i, r in enumerate(recall_metrics):
            #    recall_x_iou[i] += bools[:r].any(dim=0)
            max_sim_for_window = list()
            for idx, proposal_window in enumerate(prop):
                if sim.shape[0] >= proposal_window[1]:
                    max_sim_for_window.append(
                        float(sim[proposal_window[0].int():proposal_window[1].int()].topk(TOP_K_FRAMES)[0].mean()))

            vals, inds = torch.tensor(max_sim_for_window).sort(descending=True)
            for idx, proposal_window in enumerate(prop):
                if (proposal_window[0] <= gt_grounding[0]) and (proposal_window[1] >= gt_grounding[1]):
                    if idx in inds[0:10]:
                        inside_top10_proposals += 1
                    if idx in inds[0:20]:
                        inside_top20_proposals += 1
                    if idx in inds[0:50]:
                        inside_top50_proposals += 1

            nm_proposals.append(idx)

        acc_top10 = inside_top10_proposals / nm_queries_run
        acc_top20 = inside_top20_proposals / nm_queries_run
        acc_top50 = inside_top50_proposals / nm_queries_run
        avg_nm_proposals = np.average(nm_proposals)

        print(f"Acc for TopK frames: {TOP_K_FRAMES}, Top10 proposals is: {acc_top10 * 100:.2f}%")
        print(f"Acc for TopK frames: {TOP_K_FRAMES}, Top20 proposals is: {acc_top20 * 100:.2f}%")
        print(f"Acc for TopK frames: {TOP_K_FRAMES}, Top50 proposals is: {acc_top50 * 100:.2f}%")

        grid[f"TopF_{TOP_K_FRAMES}_WL_{WINDOW_LENGTH}"] = {'Top10': acc_top10, 'Top20': acc_top20, 'Top50': acc_top50,
                                                         'avg_nm_prop': avg_nm_proposals}
    except Exception as e:
        print(e)
try:
    for key in grid.keys():
        print(key)
        for k, v in grid[key].items():
            print(f'{k}: {v}')
except Exception as e:
    print(e)

with open('/nfs/data3/goldhofer/grid2.json', 'w') as fp:
    json.dump(grid, fp)
