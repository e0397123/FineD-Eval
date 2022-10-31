import torch


def device():
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def to_device(d, device_=None):
    if device_ is None:
        device_ = device()
    for k in d:
        if type(d[k]) == dict:
            d[k] = to_device(d[k], device_)
        elif type(d[k]) == torch.Tensor:
            d[k] = d[k].to(device_)
    return d


def max_ll(log_probs, mask):
    return torch.max(log_probs.masked_fill(~mask, -1e9), dim=-1)[0]


def marginal_ll(log_probs, mask):
    return torch.logsumexp(log_probs.masked_fill(~mask, -1e9), dim=-1)

def to_list(tensor):
    return tensor.detach().cpu().tolist()


_G = "Ġ"


def special_token_ids(tokenizer):
    return set(
        {
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.sep_token,
            tokenizer.cls_token,
            tokenizer.pad_token,
            tokenizer.mask_token,
        }
    )

def freeze(args, k):
    if "head" in k:
        return args.freeze_heads
    return args.freeze_transformer