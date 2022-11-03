import h5py
import torch
from torchmetrics.classification import BinaryRecall, BinaryPrecision
from tqdm import tqdm
from utils_benchmark import _get_movies_durations, _create_mask, compute_proposals
import traceback
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
TOP_K_PROPOSALS = [1,5,10]
rec = BinaryRecall()
prec = BinaryPrecision()

for WINDOW_LENGTH in [120,30]:
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
        recall_metrics = torch.zeros((len(annotations_keys),len(TOP_K_PROPOSALS)))
        precision_metrics = torch.zeros((len(annotations_keys), len(TOP_K_PROPOSALS)))

        # Computer performance
        for idx,k in tqdm(enumerate(annotations_keys[-500:-2])):
            try:
                movie = test_data[k]['movie']
                prop = proposals[movie]

                gt_grounding = torch.tensor(test_data[k]['ext_timestamps'])
                nm_queries_run += 1
                l_feat = torch.tensor(lang_feats[k], dtype=torch.float)[None, :]

                try:
                    p_feats = proposals_features[movie]
                except:
                    v_feat = torch.tensor(video_feats[movie], dtype=torch.float)
                    p_feats = v_feat
                    proposals_features[movie] = p_feats

                v_feat_shifted_forward = p_feats[int((WINDOW_LENGTH * FPS)/2):, :]
                v_feat_shifted_backwards = p_feats[:-int((WINDOW_LENGTH * FPS)), :]

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

                _, inds = torch.sort(sim,descending=True)
                preds = torch.zeros_like(inds)
                targets = torch.zeros_like(inds)

                target_proposal_bounds = torch.tensor([torch.floor(gt_grounding[0] / WINDOW_LENGTH),
                                                       torch.floor(gt_grounding[1] / WINDOW_LENGTH)]).unique()
                targets[target_proposal_bounds.long()] = 1

                for idk in range(len(TOP_K_PROPOSALS)):
                    preds[inds[0:TOP_K_PROPOSALS[idk]]] = 1
                    recall_metrics[idx,idk] = rec(preds, targets)
                    precision_metrics[idx, idk] = prec(preds, targets)

            except Exception:
                print(traceback.format_exc())

        mean_recall = recall_metrics.mean(axis=0)
        mean_precision = precision_metrics.mean(axis=0)

        for idk in range(len(TOP_K_PROPOSALS)):
            print(f"WL: {WINDOW_LENGTH} R@{TOP_K_PROPOSALS[idk]} = {mean_recall[idk]}")
            print(f"WL: {WINDOW_LENGTH} Precision@{TOP_K_PROPOSALS[idk]} = {mean_precision[idk]}\n")

    except Exception as e:
        print(e)