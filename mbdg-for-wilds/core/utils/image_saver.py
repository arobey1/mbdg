from torchvision.utils import save_image
import os
import torch

class ImageSaver:

    def __init__(self, args, train_ldr, val_ldr, test_ldr, G=None, delta_dim=8):
        self._args = args
        self._train_ldr = train_ldr
        self._val_ldr = val_ldr
        self._test_ldr = test_ldr
        self._G = G
        self._delta_dim = delta_dim
        self._num_imgs = 32

        self._root = os.path.join(args.results_path, 'images', args.train_alg)
        os.makedirs(self._root, exist_ok=True)

        self._mb_root = os.path.join(self._root, 'model-based')
        os.makedirs(self._mb_root, exist_ok=True)

    @torch.no_grad()
    def save_images(self):

        root = os.path.join(self._root, 'clean')
        os.makedirs(root, exist_ok=True)

        def save_clean(loader, name):
            imgs = next(iter(loader))[0][:self._num_imgs]
            save_image(imgs, os.path.join(root, f'{name}.png'))
            return imgs

        def save_model_based(imgs, name):
            delta = torch.randn(imgs.size(0), self._delta_dim, 1, 1).cuda()
            mb_images = self._G(imgs.cuda(), delta)
            save_image(mb_images, os.path.join(root, f'{name}-model-based.png'))

        train_imgs = save_clean(self._train_ldr, 'train')
        if self._G is not None:
            save_model_based(train_imgs, 'train')
            self.__save_indiv_model_based(train_imgs, 'train')

        if self._val_ldr is not None:
            save_clean(self._val_ldr, 'val')
        save_clean(self._test_ldr, 'test')

    @torch.no_grad()
    def __save_indiv_model_based(self, imgs, name, num_samples=8):

        def get_mb_image(img, reuse=False):
            # delta = 3 * torch.rand(1, self._delta_dim, 1, 1).cuda()
            delta = torch.randn(1, self._delta_dim, 1, 1).cuda()
            return self._G(img.unsqueeze(0), delta, reuse_style=reuse).squeeze()
            
        ones = torch.ones_like(imgs[0]).cuda()
        for img_idx, img in enumerate(imgs):
            img = img.cuda()

            same_style_mb = get_mb_image(img, reuse=True)
            mb_imgs = [get_mb_image(img) for _ in range(num_samples)]

            img_dir = os.path.join(self._mb_root, 'trials', f'trial-{img_idx}')
            os.makedirs(img_dir, exist_ok=True)
            save_image(img, os.path.join(img_dir, 'orig.png'))
            for idx_, mb_img_ in enumerate(mb_imgs):
                save_image(mb_img_, os.path.join(img_dir, f'mb-{idx_}.png'))

            # all_imgs = [img, ones, same_style_mb, ones] + mb_imgs
            all_imgs = [img, ones] + mb_imgs

            row = torch.cat(all_imgs, dim=-1)
            save_image(row, os.path.join(self._mb_root, f'img-{img_idx}.png'))
            
            if img_idx == 20:
                return

    def sample(self):
        x_a = next(iter(self._train_ldr))[0][:self._num_imgs].cuda()
        s_b1 = torch.randn(x_a.size(0), self._delta_dim, 1, 1).cuda()
        s_b2 = torch.randn(x_a.size(0), self._delta_dim, 1, 1).cuda()
        x_a_recon, x_ab1, x_ab2 = [], [], []
        for i in range(x_a.size(0)):
            c_a, s_a_fake = self._G.gen_A.encode(x_a[i].unsqueeze(0))
            x_a_recon.append(self._G.gen_A.decode(c_a, s_a_fake))
            x_ab1.append(self._G.gen_B.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self._G.gen_B.decode(c_a, s_b2[i].unsqueeze(0)))
        x_a_recon = torch.cat(x_a_recon)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)

        out = torch.cat([x_a, x_a_recon, x_ab1, x_ab2], dim=0)
        root = os.path.join(self._root, 'munit-minic')
        os.makedirs(root, exist_ok=True)
        save_image(out, os.path.join(root, 'recon.png'))
                

    def sample_vectorized(self):
        x_a = next(iter(self._train_ldr))[0][:self._num_imgs].cuda()
        s_b1 = torch.randn(x_a.size(0), self._delta_dim, 1, 1).cuda()
        s_b2 = torch.randn(x_a.size(0), self._delta_dim, 1, 1).cuda()

        c_a, s_a_fake = self._G.gen_A.encode(x_a)
        x_a_recon = self._G.gen_A.decode(c_a, s_a_fake)
        x_ab1 = self._G.gen_B.decode(c_a, s_b1)
        x_ab2 = self._G.gen_B.decode(c_a, s_b2)

        out = torch.cat([x_a, x_a_recon, x_ab1, x_ab2], dim=0)
        root = os.path.join(self._root, 'munit-minic')
        os.makedirs(root, exist_ok=True)
        save_image(out, os.path.join(root, 'recon-vectorized.png'))



            
