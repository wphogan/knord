import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture


def cluster_remap_preds(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = hungarian(w.max() - w)
    y_pred_remapped = np.zeros(y_pred.size, dtype=np.int64)
    for i in range(y_pred.size):
        mapped_pred = col_ind[y_pred[i]]
        y_pred_remapped[i] = mapped_pred

    accuracy = np.mean(y_pred_remapped == y_true)

    print(f'After remap, ACC: {accuracy}')

    return y_pred_remapped.tolist()


class Confusion(object):
    """
    column of confusion matrix: predicted index
    row of confusion matrix: target index
    """

    def __init__(self, k, normalized=False):
        super(Confusion, self).__init__()
        self.k = k
        self.conf = torch.LongTensor(k, k)
        self.normalized = normalized
        self.reset()

    def reset(self):
        self.conf.fill_(0)
        self.gt_n_cluster = None

    def cuda(self):
        self.conf = self.conf.cuda()

    def add(self, output, target):
        output = output.squeeze()
        target = target.squeeze()
        assert output.size(0) == target.size(0), \
            'number of targets and outputs do not match'
        if output.ndimension() > 1:  # it is the raw probabilities over classes
            assert output.size(1) == self.conf.size(0), \
                'number of outputs does not match size of confusion matrix'

            _, pred = output.max(1)  # find the predicted class
        else:  # it is already the predicted class
            pred = output
        indices = (target * self.conf.stride(0) + pred.squeeze_().type_as(target)).type_as(self.conf)
        ones = torch.ones(1).type_as(self.conf).expand(indices.size(0))
        self._conf_flat = self.conf.view(-1)
        self._conf_flat.index_add_(0, indices, ones)

    def classIoU(self, ignore_last=False):
        confusion_tensor = self.conf
        if ignore_last:
            confusion_tensor = self.conf.narrow(0, 0, self.k - 1).narrow(1, 0, self.k - 1)
        union = confusion_tensor.sum(0).view(-1) + confusion_tensor.sum(1).view(-1) - confusion_tensor.diag().view(-1)
        acc = confusion_tensor.diag().float().view(-1).div(union.float() + 1)
        return acc

    def recall(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFN = self.conf[i, :].sum().item()
        if TPuFN == 0:
            return 0
        return float(TP) / TPuFN

    def precision(self, clsId):
        i = clsId
        TP = self.conf[i, i].sum().item()
        TPuFP = self.conf[:, i].sum().item()
        if TPuFP == 0:
            return 0
        return float(TP) / TPuFP

    def f1score(self, clsId):
        r = self.recall(clsId)
        p = self.precision(clsId)
        print("classID:{}, precision:{:.4f}, recall:{:.4f}".format(clsId, p, r))
        if (p + r) == 0:
            return 0
        return 2 * float(p * r) / (p + r)

    def acc(self):
        TP = self.conf.diag().sum().item()
        total = self.conf.sum().item()
        if total == 0:
            return 0
        return float(TP) / total

    def optimal_assignment(self, gt_n_cluster=None, assign=None):
        if assign is None:
            mat = -self.conf.cpu().numpy()  # hungaian finds the minimum cost
            r, assign = hungarian(mat)
        self.conf = self.conf[:, assign]
        self.gt_n_cluster = gt_n_cluster
        return assign

    def show(self, width=6, row_labels=None, column_labels=None):
        print("Confusion Matrix:")
        conf = self.conf
        rows = self.gt_n_cluster or conf.size(0)
        cols = conf.size(1)
        if column_labels is not None:
            print(("%" + str(width) + "s") % '', end='')
            for c in column_labels:
                print(("%" + str(width) + "s") % c, end='')
            print('')
        for i in range(0, rows):
            if row_labels is not None:
                print(("%" + str(width) + "s|") % row_labels[i], end='')
            for j in range(0, cols):
                print(("%" + str(width) + ".d") % conf[i, j], end='')
            print('')

    def conf2label(self):
        conf = self.conf
        gt_classes_count = conf.sum(1).squeeze()
        n_sample = gt_classes_count.sum().item()
        gt_label = torch.zeros(n_sample)
        pred_label = torch.zeros(n_sample)
        cur_idx = 0
        for c in range(conf.size(0)):
            if gt_classes_count[c] > 0:
                gt_label[cur_idx:cur_idx + gt_classes_count[c]].fill_(c)
            for p in range(conf.size(1)):
                if conf[c][p] > 0:
                    pred_label[cur_idx:cur_idx + conf[c][p]].fill_(p)
                cur_idx = cur_idx + conf[c][p]
        return gt_label, pred_label

    def clusterscores(self):
        target, pred = self.conf2label()
        NMI = normalized_mutual_info_score(target, pred)
        ARI = adjusted_rand_score(target, pred)
        AMI = adjusted_mutual_info_score(target, pred)
        return {'NMI': NMI, 'ARI': ARI, 'AMI': AMI}


def get_GMM(all_features, all_labels, num_classes, ARGS):
    all_features = all_features.numpy()
    all_features = preprocessing.normalize(all_features)
    if ARGS.pca > 0:
        print(f'Conduct PCA, reduce dimenstion to {ARGS.pca}')
        _pca = PCA(n_components=ARGS.pca)
        all_features = _pca.fit_transform(all_features)
    print('Clustering with GMM...')
    # Perform kmean clustering
    confusion = Confusion(num_classes)
    clustering_model = GaussianMixture(n_components=num_classes, n_init=100, covariance_type='tied', warm_start=True,
                                       verbose=1)
    clustering_model.fit(all_features)
    cluster_assignment = clustering_model.predict(all_features)
    scores = clustering_model.predict_proba(all_features)

    if all_labels is None:
        return scores

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)

    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)

    print("Clustering iterations:{}, ACC:{:.3f}".format(clustering_model.n_iter_, confusion.acc()))
    print('Clustering scores:', confusion.clusterscores())

    return scores
