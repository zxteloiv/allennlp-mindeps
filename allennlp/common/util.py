"""
Various utilities that don't fit anywhere else.
"""
from datetime import timedelta
import logging
import sys
from typing import Generator, TypeVar, Union

import torch
import torch.distributed as dist

try:
    import resource
except ImportError:
    # resource doesn't exist on Windows systems
    resource = None  # type: ignore

logger = logging.getLogger(__name__)

T = TypeVar("T")
ContextManagerFunctionReturnType = Generator[T, None, None]


def peak_cpu_memory() -> dict[int, int]:
    """
    Get peak memory usage for each worker, as measured by max-resident-set size:

    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size

    Only works on OSX and Linux, otherwise the result will be 0.0 for every worker.
    """
    if resource is None or sys.platform not in ("linux", "darwin"):
        peak_bytes = 0
    else:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            # On OSX the result is in bytes.
            peak_bytes = peak
        else:
            # On Linux the result is in kilobytes.
            peak_bytes = peak * 1_024

    if is_distributed():
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes])
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0]) for _ in range(world_size)]

        # If the backend is 'nccl', this means we're training on GPUs, so these tensors
        # need to be on GPU.
        if dist.get_backend() == "nccl":
            peak_bytes_tensor = peak_bytes_tensor.cuda()
            gather_results = [x.cuda() for x in gather_results]

        dist.all_gather(gather_results, peak_bytes_tensor)

        results_dict: dict[int, int] = {}
        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])

        return results_dict
    else:
        return {0: peak_bytes}


def peak_gpu_memory() -> dict[int, int]:
    """
    Get the peak GPU memory usage in bytes by device.

    # Returns

    `dict[int, int]`
        Keys are device ids as integers.
        Values are memory usage as integers in bytes.
        Returns an empty `dict` if GPUs are not available.
    """
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()

    results_dict: dict[int, int] = {}
    if is_distributed():
        # If the backend is not 'nccl', we're training on CPU.
        if dist.get_backend() != "nccl":
            return {}

        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes], device=device)
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0], device=device) for _ in range(world_size)]

        dist.all_gather(gather_results, peak_bytes_tensor)

        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])
    else:
        results_dict = {0: torch.cuda.max_memory_allocated()}

    # Reset peak stats.
    torch.cuda.reset_max_memory_allocated(device)

    return results_dict


def int_to_device(device: Union[int, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device < 0:
        return torch.device("cpu")
    return torch.device(device)


def is_distributed() -> bool:
    """
    Checks if the distributed process group is available and has been initialized
    """
    return dist.is_available() and dist.is_initialized()


def is_global_primary() -> bool:
    """
    Checks if the distributed process group is the global primary (rank = 0).
    If the distributed process group is not available or has not been initialized,
    this trivially returns `True`.
    """
    if not is_distributed():
        return True
    else:
        return dist.get_rank() == 0


def sanitize_wordpiece(wordpiece: str) -> str:
    """
    Sanitizes wordpieces from BERT, RoBERTa or ALBERT tokenizers.
    """
    if wordpiece.startswith("##"):
        return wordpiece[2:]
    elif wordpiece.startswith("Ġ"):
        return wordpiece[1:]
    elif wordpiece.startswith("▁"):
        return wordpiece[1:]
    else:
        return wordpiece


def sanitize_ptb_tokenized_string(text: str) -> str:
    """
    Sanitizes string that was tokenized using PTBTokenizer
    """
    tokens = text.split(" ")
    if len(tokens) == 0:
        return text

    # Replace quotation marks and parentheses
    token_map = {
        "``": '"',
        "''": '"',
        "-lrb-": "(",
        "-rrb-": ")",
        "-lsb-": "[",
        "-rsb-": "]",
        "-lcb-": "{",
        "-rcb-": "}",
        "<s>": "",
        "</s>": "",
    }

    # Merge punctuation with previous tokens
    punct_forward = {"`", "$", "#"}
    punct_backward = {".", ",", "!", "?", ":", ";", "%", "'"}

    # Exact matches that get merged forward or backward
    em_forward = {"(", "[", "{"}
    em_backward = {"n't", "na", ")", "]", "}"}

    new_tokens: list[str] = []

    merge_fwd = False
    for i, orig_token in enumerate(tokens):
        tokens[i] = token_map[orig_token.lower()] if orig_token.lower() in token_map else orig_token
        new_token = tokens[i].lower()

        # merge_fwd was set by previous token, so it should be prepended to current token
        if merge_fwd:
            tokens[i] = tokens[i - 1] + tokens[i]

        if len(tokens[i]) == 0:
            continue

        # Special cases for `` and '', those tells us if " is the start or end of a quotation.
        # Also always merge tokens starting with ' backward and don't merge back if we just merged forward
        merge_bckwd = not merge_fwd and (
            orig_token == "''"
            or new_token in em_backward
            or new_token.startswith("'")
            or all(c in punct_backward for c in new_token)
        )
        merge_fwd = (
            orig_token == "``"
            or new_token in em_forward
            or all(c in punct_forward for c in new_token)
        )

        if merge_bckwd and new_tokens:
            new_tokens[-1] += tokens[i]
        elif not new_tokens or not merge_fwd or i == len(tokens) - 1:
            new_tokens.append(tokens[i])

    return " ".join(new_tokens)


def find_open_port() -> int:
    """
    Find a random open port on local host.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # Passes 0 means find any open port.
        # See https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number
        sock.bind(("", 0))
        return sock.getsockname()[1]


def format_timedelta(td: timedelta) -> str:
    """
    Format a timedelta for humans.
    """
    if td.days > 1:
        return f"{td.days} days"
    elif td.days > 0:
        return f"{td.days} day"
    else:
        hours, remainder = divmod(td.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 1:
            return f"{hours} hours"
        elif hours > 0:
            return f"{hours} hour, {minutes} mins"
        else:
            return f"{minutes} mins"


def format_size(size: int) -> str:
    """
    Format a size (in bytes) for humans.
    """
    GBs = size / (1024 * 1024 * 1024)
    if GBs >= 10:
        return f"{int(round(GBs, 0))}G"
    if GBs >= 1:
        return f"{round(GBs, 1):.1f}G"
    MBs = size / (1024 * 1024)
    if MBs >= 10:
        return f"{int(round(MBs, 0))}M"
    if MBs >= 1:
        return f"{round(MBs, 1):.1f}M"
    KBs = size / 1024
    if KBs >= 10:
        return f"{int(round(KBs, 0))}K"
    if KBs >= 1:
        return f"{round(KBs, 1):.1f}K"
    return f"{size}B"


def nan_safe_tensor_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result

