import yaml
import torch
import torch.nn as nn
from apex import amp

from models.munit.core.networks import AdaINGen


def load_model(model_path, half_prec, config_path, reverse=False):
    """Load MUNIT model and initialize with half-precision
    if args.half_precision flag is set."""

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    G = MUNITModelOfNatVar(model_path, reverse=reverse, config=config).cuda()
    if half_prec is True:
        G = amp.initialize(G, opt_level='O2', verbosity=0).half()
    return G


class MUNITModelOfNatVar(nn.Module):
    def __init__(self, fname: str, reverse: bool, config: str):
        """Instantiantion of pre-trained MUNIT model.
        
        Params:
            fname: File name of trained MUNIT checkpoint file.
            reverse: If True, returns model mapping from domain A-->B.
                otherwise, model maps from B-->A.
            config: Configuration .yaml file for MUNIT.
        """

        super(MUNITModelOfNatVar, self).__init__()

        self._config = config
        self._fname = fname
        self._reverse = reverse
        self._gen_A, self._gen_B = self.__load()
        self.delta_dim = self._config['gen']['style_dim']

    @property
    def gen_A(self):
        return self._gen_A

    @property
    def gen_B(self):
        return self._gen_B

    # def forward(self, x, delta, reuse_style=False):
    #     """Forward pass through MUNIT model of natural variation."""

    #     orig_content, orig_style = self._gen_A.encode(x)
    #     orig_content = orig_content.clone().detach().requires_grad_(False)

    #     if reuse_style is True:
    #         delta = orig_style.clone().detach().requires_grad_(True)

    #     if float(torch.rand(1)) > 0.5:
    #         return self._gen_B.decode(orig_content, delta)
    #     else:
    #         return self._gen_A.decode(orig_content, delta)

    def forward(self, x, delta, reuse_style=False):

        orig_content, orig_style = self._gen_A.encode(x)
        orig_content = orig_content.clone().detach().requires_grad_(False)
        return self._gen_B.decode(orig_content, delta)


    def __load(self):
        """Load MUNIT model from file."""

        def load_munit(fname, letter):
            gen = AdaINGen(self._config[f'input_dim_{letter}'], self._config['gen'])
            gen.load_state_dict(torch.load(fname)[letter])
            return gen.eval()

        gen_A = load_munit(self._fname, 'a')
        gen_B = load_munit(self._fname, 'b')

        if self._reverse is False:
            return gen_A, gen_B     # original order
        return gen_B, gen_A         # reversed order
        