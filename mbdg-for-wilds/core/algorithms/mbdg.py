import torch
import torch.nn.functional as F

class MBDG:

    def __init__(self, model, G, criterion, args):
        self._model = model
        self._G = G
        self._criterion = criterion
        self._num_steps = args.mbdg_num_steps
        self._dual_step_size = torch.tensor(args.mbdg_dual_step_size)
        self._gamma = torch.tensor(args.mbdg_gamma)
        self._delta_dim = args.delta_dim
        self._fname = (f'mbdg-lam-dist-{args.mbst_static_lam_dist}-lam-grad' +
                f'-{args.mbdg_static_lam_grad}-num-steps-{args.mbdg_num_steps}' + 
                f'-lr-{args.lr}-dual-step-{self._dual_step_size}-gamma-{self._gamma}' + 
                f'-trial-{args.trial_index}')

    def __call__(self, images, labels, dual_var):
        return self.step(images, labels, dual_var)

    @property
    def fname(self):
        return self._fname

    @staticmethod
    def relu(x):
        return x if x > 0 else torch.tensor(0.0).cuda()

    @staticmethod
    def kl_div(dist1, dist2):
        return F.kl_div(torch.log(dist2), dist1, reduction='batchmean')

    def step(self, imgs, labels, dual_var):

        clean_output = self._model(imgs)
        pred = clean_output.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        clean_loss = self._criterion(clean_output, labels)

        dist_reg = torch.tensor(0.0).cuda()
        for _ in range(self._num_steps):
            with torch.no_grad():
                delta = torch.randn(imgs.size(0), self._delta_dim, 1, 1).cuda().requires_grad_(False)
                mb_images = self._G(imgs, delta)
            mb_output = F.softmax(self._model(mb_images), dim=1)
            dist_reg += self.kl_div(F.softmax(clean_output, dim=1), mb_output)
        
        loss = clean_loss + dual_var * dist_reg

        return loss, correct, dist_reg

    def dual_step(self, dual_var, reg_term):
        return self.relu(dual_var + self._dual_step_size * (reg_term - self._gamma))

    
        



        
        
