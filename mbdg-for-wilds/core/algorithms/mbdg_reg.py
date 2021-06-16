import torch
import torch.nn.functional as F

class MBDG_Reg:

    def __init__(self, model, G, criterion, args):
        self._model = model
        self._G = G
        self._criterion = criterion
        self._num_steps = args.mbdg_num_steps
        self._lam_dist = args.mbdg_static_lam_dist
        self._lam_grad = args.mbdg_static_lam_grad
        self._delta_dim = args.delta_dim
        self._fname = (f'mbdg-reg-lam-dist-{args.mbdg_static_lam_dist}-lam-grad' +
                f'-{args.mbdg_static_lam_grad}-num-steps-{args.mbdg_num_steps}' + 
                f'-lr-{args.lr}-trial-{args.trial_index}')

    def __call__(self, images, labels):
        return self.step(images, labels)

    @property
    def fname(self):
        return self._fname

    def step(self, imgs, labels):

        clean_output = self._model(imgs)
        pred = clean_output.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        clean_loss = self._criterion(clean_output, labels)

        dist_reg = torch.tensor(0.0).cuda()
        for _ in range(self._num_steps):

            # distance regularization
            with torch.no_grad():
                delta = torch.randn(imgs.size(0), self._delta_dim, 1, 1).cuda().requires_grad_(False)
                mb_images = self._G(imgs, delta)
            mb_output = F.softmax(self._model(mb_images), dim=1)
            dist_reg += self.kl_div(F.softmax(clean_output, dim=1), mb_output)        

        loss = clean_loss + self._lam_dist * dist_reg

        return loss, correct

    @staticmethod
    def kl_div(dist1, dist2):
        return F.kl_div(torch.log(dist2), dist1, reduction='batchmean')
        



        
        
