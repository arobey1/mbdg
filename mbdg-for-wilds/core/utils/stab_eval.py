import torch
import torch.nn.functional as F

import core.utils.dist_utils as dist_utils
from core.utils.meters import VarTracker

class StabEval:

    def __init__(self, model, G, delta_dim, num_trials=1):
        self._model = model
        self._G = G
        self._delta_dim = delta_dim
        self._num_trials = num_trials

    @torch.no_grad()
    def evaluate(self, loader):

        dist_tracker = VarTracker()

        self._model.eval()
        for (imgs, labels) in loader:
            imgs = imgs.cuda()

            for idx in range(self._num_trials):
                delta = torch.randn(imgs.size(0), self._delta_dim, 1, 1).cuda()
                mb_imgs = self._G(imgs, delta)

                clean_logits = F.softmax(self._model(imgs), dim=1)
                mb_logits = F.softmax(self._model(mb_imgs), dim=1)
                dist = self.kl_div(clean_logits, mb_logits)

                all_dists = dist_utils.multi_varsize_all_gather(dist, cat_dim=0)
                dist_tracker.update(all_dists.cpu().numpy())

        return dist_tracker.stacked()

    @staticmethod
    def kl_div(dist1, dist2):
        return torch.sum(F.kl_div(torch.log(dist2), dist1, reduction='none'), dim=1).unsqueeze(1)
        
