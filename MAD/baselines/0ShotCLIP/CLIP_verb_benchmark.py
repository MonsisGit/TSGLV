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
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from collections import Counter

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

WINDOW_LENGTH = 60
for TOP_K_FRAMES in [10,20]:
    try:

        nm_queries_run = 0

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
        for k in tqdm(annotations_keys[10]):
            movie = test_data[k]['movie']
            prop = proposals[movie]
            inside_top10_proposals = 0
            # windows_idx = torch.round(prop * FPS).int()
            # windows_idx = torch.arange(0, movies_durations[movie], 1)
            gt_grounding = torch.tensor(test_data[k]['ext_timestamps'])
            # Get movie features and sentence features
            l_feat = torch.tensor(lang_feats[k], dtype=torch.float)[None, :]

            sentence_tokens = nltk.word_tokenize(test_data[k]["sentence"].lower())
            sentence_text = nltk.Text(sentence_tokens)
            sentence_tags = nltk.pos_tag(sentence_text, tagset='universal')
            sentence_counted = Counter(sentence_tags)
            nm_verbs_in_sentence = len([t[0] for t in sentence_counted if t[1] == "VERB"])
            nm_nouns_in_sentence = len([t[0] for t in sentence_counted if t[1] == "NOUN"])

            try:
                p_feats = proposals_features[movie]
            except:
                v_feat = torch.tensor(video_feats[movie], dtype=torch.float)
                # p_feats = _compute_proposals_feats(v_feat, windows_idx)
                p_feats = v_feat
                proposals_features[movie] = p_feats

            sim = cosine_similarity(l_feat, p_feats)
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
                        inside_top10_proposals = 1

            grid[f"{k}_{movie}"] = {'nm_verbs': nm_verbs_in_sentence, 'nm_nouns': nm_nouns_in_sentence,
                                    'inside_top10_proposals': inside_top10_proposals}


    except Exception as e:
        print(e)
    verb_count = np.zeros(shape=(20,2))
    noun_count = np.zeros(shape=(20,2))
    for g in [grid[k] for k in grid.keys()]:
        try:
            verb_count[g["nm_verbs"],g["inside_top10_proposals"]]+=1
            noun_count[g["nm_nouns"],g["inside_top10_proposals"]]+=1
        except Exception as e:
            print(e)

    acc_per_nm_verbs = np.nan_to_num(verb_count[:,1]/verb_count.sum(axis=1))
    acc_per_nm_nouns = np.nan_to_num(noun_count[:,1]/noun_count.sum(axis=1))

    plt.figure()
    plt.plot(acc_per_nm_verbs)
    plt.xticks([k*2 for k in range(int(verb_count.shape[0]/2)+1)])
    plt.title("Acc per number of verbs in query")
    plt.xlabel("number of verbs in query")
    plt.ylabel(f"acc [%] (TopP=10,TopF={TOP_K_FRAMES}, Wl = 60)")
    plt.savefig(f"/nfs/data3/goldhofer/clip_acc_per_nm_verbs_new{TOP_K_FRAMES}.jpg")
    plt.close()
    plt.clf()

    plt.figure()
    plt.plot(acc_per_nm_nouns)
    plt.xticks([k*2 for k in range(int(noun_count.shape[0]/2)+1)])
    plt.title("Acc per number of nouns in query")
    plt.xlabel("number of nouns in query")
    plt.ylabel(f"acc [%] (TopP=10,TopF={TOP_K_FRAMES} Wl = 60)")
    plt.savefig(f"/nfs/data3/goldhofer/clip_acc_per_nm_nouns_new{TOP_K_FRAMES}.jpg")

    with open(f'/nfs/data3/goldhofer/verb_noun_acc_benchmark_{TOP_K_FRAMES}.json', 'w') as fp:
        json.dump(grid, fp)
