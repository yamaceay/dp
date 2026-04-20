from typing import List, Optional, Union

import torch
def resolve_device(preferred: Optional[Union[str, int]] = None) -> str:
    device_selected: Optional[str] = None
    devices = available_devices()
    if len(devices) == 0:
        raise RuntimeError("No available devices found")
    if preferred is None:
        device_selected = devices[0]
    if isinstance(preferred, str):
        preferred = preferred.lower()
        if preferred not in {"cpu", "mps"} and not preferred.startswith("cuda"):
            raise ValueError(f"Invalid device string: {preferred}")
    elif isinstance(preferred, int):
        if preferred < 0:
            raise ValueError(f"Invalid CUDA device index: {preferred}")
        preferred = f"cuda:{preferred}"
    for dev in devices:
        if dev == preferred or (preferred == "cuda" and dev.startswith("cuda")):
            device_selected = dev
            break
    if device_selected is None:
        raise ValueError(f"Preferred device {preferred} is not available. Available devices: {devices}")
    print(f"Selected device: {device_selected}")
    return device_selected

def available_devices() -> List[str]:
    devices: List[str] = []
    if torch.cuda.is_available():
        n_cuda = torch.cuda.device_count()
        devices.extend([f"cuda:{i}" for i in range(n_cuda)])
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices