import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import multiprocessing
import os

def setup_dist_training(args):
    """Setup distributed training with command line arguments."""

    is_master, is_rank0 = whoami(args)
    args.world_size = env_world_size()
    args.rank = env_rank()
    setup_dist_backend(args)
    sync_processes(args)


def whoami(args):
    """Determines if current rank is master and/or local rank 0.
    
    Params:
        args: Command line arguments for main.py.
    """

    is_master = (not args.distributed) or (env_rank() == 0)
    is_rank0 = args.local_rank == 0

    return is_master, is_rank0

def env_world_size(): 
    """World size for distributed training.
    Is set in torch.distributed.launch as args.nproc_per_node * args.nnodes.
    For example, when running on 1 node with 4 GPUs per node, the world size is 4.
    see: https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py"""

    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return 1

def env_rank(): 
    """Local rank of each GPU used in distributed training.
    Is set in torch.distributed.launch as args.nproc_per_node * args.node_rank + local_rank
    see: https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py"""

    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return 1

def setup_dist_backend(args, set_threads=False, thread_choice=None):
    """Sets up backend/environment for distributed training.
    Params:
        args: Command line args for main.py.
        thread_choice: How to choose number of OMP threads used.
    """

    # assumes all data will have (roughly) the same dimensions
    cudnn.benchmark = True

    # choose environment variable OMP_NUM_THREADS
    # see: https://github.com/pytorch/pytorch/pull/22501
    if set_threads is True:
        if thread_choice is None:
            os.environ['OMP_NUM_THREADS'] = str(1)
        elif thread_choice == 'torch_threads':
            os.environ['OMP_NUM_THREADS'] = str(torch.get_num_threads())
        elif thread_choice == 'multiproc':
            n_threads = (int)(multiprocessing.cpu_count() / os.environ['WORLD_SIZE'])
            os.environ['OMP_NUM_THREADS'] = str(n_threads)

    if args.distributed is True:
        if args.local_rank == 0:
            print('Setting up distributed process group...')

        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend=args.dist_backend, 
            init_method=args.dist_url, 
            world_size=env_world_size()
        )

        # make sure there's no mismatch between world sizes
        assert(env_world_size() == torch.distributed.get_world_size())
        print(f"\tSuccess on process {args.local_rank}/{torch.distributed.get_world_size()}")
        
def reduce_tensor(tensor): 
    return sum_tensor(tensor) / env_world_size()

def sum_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt

def sync_processes(args):
    """Perform a simple reduce operation to sync all processes.
    
    Params:
        args: Command line args for main.py.
    """

    tensor = torch.tensor([1.0]).float().cuda()
    rt = sum_tensor(tensor)

    if args.local_rank == 0:
        print(f'Gave tensor = {tensor.item()} to each process.  Summed results: {rt.item()}')

def multi_varsize_all_gather(tensor, cat_dim=None, pad_val=-100_000):
    """We are given a tensor {tensor}, where the shape of the tensor may vary
    across distributed processes.  This function performs all_gather on this 
    tensor and avoids errors due to mismatched sizes.
    
    Params:
        tensor: Input tensor that we want to use for all_gather.
        cat_dim: Dimension to concatanate gathered tensors.  If cat_dim=None,
            a list of gathered tensors will be returned.
        pad_val: Value to pad the final all_gather.  Note that this value should
            be chosen so as not to conflict with a value that would appear 
            ordinarily in {tensor}.  So for example, in binary classification where
            a prediction consists of a vector with values in [0,1], a reasonable 
            choice for {pad_val} would be any value not in [0, 1] (e.g. -100_000).
    """

    int_kwargs = {'dtype': torch.int64, 'device': tensor.device}
    float_kwargs = {'dtype': torch.float64, 'device': tensor.device}
    world_size = env_world_size()

    local_size = torch.tensor(tensor.shape, **int_kwargs)
    num_dims = local_size.numel()
    size_list = [torch.zeros(num_dims, **int_kwargs) for _ in range(world_size)]

    # gather list of sizes of tensors
    dist.all_gather(size_list, local_size)

    # find the max dimensions of any of the tensors
    size_mat = torch.stack(size_list, dim=1)
    max_sizes = torch.max(size_mat, dim=1).values

    # create padded tensor on each process for all_gather
    pad_tensor = torch.ones(*max_sizes, **float_kwargs) * pad_val
    ind = [slice(0, d) for d in tensor.shape]
    pad_tensor[ind] = tensor

    # put all padded tensors into a list and all_gather on padded tensors
    pad_list = [torch.zeros(*max_sizes, **float_kwargs) for _ in range(world_size)]
    dist.all_gather(pad_list, pad_tensor)

    # create a mask for non-padded values
    true_vals = [t != pad_val for t in pad_list]

    # remove flattened values; results is flattened
    final_flattened = [t[bools] for t, bools in zip(pad_list, true_vals)]

    # make list of the sizes of original tensors and reshape flattened tensors
    int_size_list = [tuple(shape.cpu().numpy()) for shape in size_list]
    final_ls = [t.reshape(*shape) for (t, shape) in zip(final_flattened, int_size_list)]

    if cat_dim is not None:
        return torch.cat(final_ls, dim=cat_dim)
    return final_ls