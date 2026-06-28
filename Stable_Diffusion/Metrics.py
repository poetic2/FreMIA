import numpy as np
from sklearn import metrics


def result(member_scores, nonmember_scores):

    member_scores = np.array(member_scores)
    nonmember_scores = np.array(nonmember_scores)

    min_score = min(member_scores.min(), nonmember_scores.min())
    max_score = max(member_scores.max(), nonmember_scores.max())

    TPR_list = []
    FPR_list = []

    TPRatFPR_1 = 0
    FPR_1_idx = 999
    TPRatFPR_01 = 0
    FPR_01_idx = 999

    total = member_scores.size + nonmember_scores.size
    max_acc = 0.0
    best_threshold = 0.0

    thresholds = np.linspace(min_score, max_score, 10000)

    for threshold in thresholds:
        acc = ((member_scores <= threshold).sum() + (nonmember_scores > threshold).sum()) / total

        TP = (member_scores <= threshold).sum()
        TN = (nonmember_scores > threshold).sum()
        FP = (nonmember_scores <= threshold).sum()
        FN = (member_scores > threshold).sum()

        TPR = TP / (TP + FN + 1e-10)
        FPR = FP / (FP + TN + 1e-10)

        if FPR_1_idx > abs(0.01 - FPR):
            FPR_1_idx = abs(0.01 - FPR)
            TPRatFPR_1 = TPR

        if FPR_01_idx > abs(0.001 - FPR):
            FPR_01_idx = abs(0.001 - FPR)
            TPRatFPR_01 = TPR

        TPR_list.append(TPR)
        FPR_list.append(FPR)

        if acc > max_acc:
            max_acc = acc
            best_threshold = threshold

    auc = metrics.auc(np.asarray(FPR_list), np.asarray(TPR_list))
    print(
        f'AUC: {auc:.6f} \t ASR: {max_acc:.6f} \t TPR@FPR=1%: {TPRatFPR_1:.6f} \t TPR@FPR=0.1%: {TPRatFPR_01:.6f}')
    print(f'Threshold: {best_threshold:.6f}')

    return auc, FPR_list, TPR_list
