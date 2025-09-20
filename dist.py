import pickle
import os
import torch
import torch.distributed as dist


def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_primary():
    return get_rank() == 0


def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()


def barrier():
    if not is_distributed():
        return
    torch.distributed.barrier()


def setup_print_for_distributed(is_primary):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_primary or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print



def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    args.lr *= max(1., args.world_size * args.batch_size / 64)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()


def all_reduce_sum(tensor):
    if not is_distributed():
        return tensor
    dim_squeeze = False
    if tensor.ndim == 0:
        tensor = tensor[None, ...]
        dim_squeeze = True
    torch.distributed.all_reduce(tensor)
    if dim_squeeze:
        tensor = tensor.squeeze(0)
    return tensor


def all_reduce_average(tensor):
    val = all_reduce_sum(tensor)
    return val / get_world_size()


def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        torch.distributed.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def all_gather_pickle(data, device):
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = [torch.tensor([0], device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device=device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def all_gather_dict(data):
    assert isinstance(data, dict)
    
    gathered_dict = {}
    for item_key in data:
        if isinstance(data[item_key], torch.Tensor):
            if is_distributed():
                data[item_key] = data[item_key].contiguous()
                tensor_list = [torch.empty_like(data[item_key]) for _ in range(get_world_size())]
                dist.all_gather(tensor_list, data[item_key])
                gathered_tensor = torch.cat(tensor_list, dim=0)
            else:
                gathered_tensor = data[item_key]
            gathered_dict[item_key] = gathered_tensor
    return gathered_dict
