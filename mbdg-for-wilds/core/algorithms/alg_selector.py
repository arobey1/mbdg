from core.algorithms.erm import ERM
from core.algorithms.mbdg_reg import MBDG_Reg
from core.algorithms.mbdg import MBDG

def alg_selector(model, criterion, G, args):

    if args.train_alg.lower() == 'erm':
        return ERM(model, criterion, args)

    elif args.train_alg.lower() == 'mbdg-reg':
        return MBDG_Reg(model, G, criterion, args)

    elif args.train_alg.lower() == 'mbdg':
        return MBDG(model, G, criterion, args)

    else:
        raise NotImplementedError(f'Algorithm {args.train_alg} is not implemented')