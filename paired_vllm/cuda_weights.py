"""Borrow resident training weights across torch versions using CUDA driver IPC.

The consumer copies into its own vLLM parameters without a host round trip.
Default transfers close mappings before acknowledging SYNC. Optional cached
mappings require the trainer to retain the exported allocations until the
consumer remaps or closes; they cache views, never the weight values.

Same-GPU consumers map the exporter's allocations directly. A consumer on a
different GPU (the resident teacher worker pulling the trainer's EMA teacher)
opens the same handles with lazy peer access and issues one device-to-device
peer copy per tensor into local staging, then feeds the staging tensors to the
stock weight loaders. The exporter's GPU may be hidden from the consumer by
CUDA_VISIBLE_DEVICES; unified addressing still resolves the mapped memory.
"""

import ctypes
import time
from contextlib import contextmanager
from math import prod

from paired_vllm.cuda_cache import _check, _driver, _Handle


def export_cuda_weights(state):
    import torch

    devices = {tensor.device for tensor in state.values()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        raise ValueError("CUDA weight export requires one resident CUDA device")
    device = next(iter(devices))
    api = _driver()
    address_range = api.cuMemGetAddressRange_v2
    address_range.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_uint64,
    ]
    address_range.restype = ctypes.c_int
    allocations, tensors = {}, {}
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        for name, tensor in state.items():
            if not tensor.is_contiguous() or tensor.numel() == 0:
                raise ValueError(
                    "CUDA weight export requires nonempty contiguous tensors"
                )
            base, size = ctypes.c_uint64(), ctypes.c_size_t()
            _check(
                address_range(ctypes.byref(base), ctypes.byref(size), tensor.data_ptr())
            )
            if base.value not in allocations:
                handle = _Handle()
                _check(api.cuIpcGetMemHandle(ctypes.byref(handle), base))
                allocations[base.value] = (bytes(handle), size.value)
            handle, size = allocations[base.value]
            tensors[name] = {
                "handle": handle,
                "allocation_bytes": size,
                "offset": tensor.data_ptr() - base.value,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
    return {
        "format": "cuda-driver-weights-v1",
        "device_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "tensors": tensors,
    }


class _ByteView:
    def __init__(self, pointer, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }


@contextmanager
def import_cuda_weights(payload):
    import torch

    if payload["format"] != "cuda-driver-weights-v1":
        raise ValueError("unknown CUDA weight format")
    if (
        str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        != payload["device_uuid"]
    ):
        raise ValueError("training and inference student must use the same GPU")
    opened, state = {}, {}
    api = _driver()
    torch.cuda.current_stream()  # Materialize the primary context before driver IPC.
    try:
        for name, item in payload["tensors"].items():
            dtype = getattr(torch, item["dtype"])
            nbytes = prod(item["shape"]) * torch.empty((), dtype=dtype).element_size()
            if item["offset"] < 0 or item["offset"] + nbytes > item["allocation_bytes"]:
                raise ValueError("weight exceeds CUDA allocation")
            handle = item["handle"]
            if len(handle) != 64:
                raise ValueError("invalid CUDA weight handle")
            if handle not in opened:
                pointer = ctypes.c_uint64()
                _check(
                    api.cuIpcOpenMemHandle(
                        ctypes.byref(pointer), _Handle.from_buffer_copy(handle), 1
                    )
                )
                opened[handle] = pointer
            view = _ByteView(opened[handle].value + item["offset"], nbytes)
            state[name] = (
                torch.as_tensor(view, device="cuda").view(dtype).reshape(item["shape"])
            )
        yield state
    finally:
        # vLLM weight loaders can enqueue asynchronous copies from these views.
        torch.cuda.synchronize()
        state.clear()
        for pointer in opened.values():
            _check(api.cuIpcCloseMemHandle(pointer))


class CudaWeightMappingCache:
    """Retain borrowed views while the exporter keeps their allocations alive."""

    def __init__(self):
        self.payload = None
        self.context = None
        self.state = None

    def get(self, payload):
        if self.context is not None and self.payload == payload:
            return self.state, True
        self.close()
        context = import_cuda_weights(payload)
        state = context.__enter__()
        self.context, self.state, self.payload = context, state, payload
        return state, False

    def close(self):
        if self.context is not None:
            context = self.context
            self.context = self.state = self.payload = None
            context.__exit__(None, None, None)


class CudaWeightPuller:
    """Pull another GPU's exported weights into local staging with peer copies.

    Handles stay open across pulls while the exporter keeps the same
    allocations alive (``cache=True``); the payload changing reopens them.
    Staging tensors are reused across pulls and aliased tensors (tied
    embeddings) share one staging buffer. Weight bytes never enter host RAM.
    """

    def __init__(self):
        self.payload = None
        self.opened = {}
        self.staging = {}

    def pull(self, payload, *, cache):
        import torch

        if payload["format"] != "cuda-driver-weights-v1":
            raise ValueError("unknown CUDA weight format")
        local = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        if local == payload["device_uuid"]:
            raise ValueError("peer pull requires the exporter to be on another GPU")
        api = _driver()
        torch.cuda.current_stream()  # Materialize the primary context before driver IPC.
        started = time.perf_counter()
        reopened = self.payload != payload
        if reopened:
            self._close_handles()
            for item in payload["tensors"].values():
                handle = item["handle"]
                if len(handle) != 64:
                    raise ValueError("invalid CUDA weight handle")
                if handle not in self.opened:
                    pointer = ctypes.c_uint64()
                    # CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS maps the exporter's
                    # device memory into this context when it lives elsewhere.
                    _check(
                        api.cuIpcOpenMemHandle(
                            ctypes.byref(pointer), _Handle.from_buffer_copy(handle), 1
                        )
                    )
                    self.opened[handle] = pointer
            self.payload = payload
        plan, aliases = {}, {}
        for name, item in payload["tensors"].items():
            dtype = getattr(torch, item["dtype"])
            shape = tuple(item["shape"])
            nbytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
            if item["offset"] < 0 or item["offset"] + nbytes > item["allocation_bytes"]:
                raise ValueError("weight exceeds CUDA allocation")
            identity = (item["handle"], item["offset"], shape, item["dtype"])
            if identity in aliases:
                plan[name] = (self.staging[aliases[identity]], None, item)
                continue
            aliases[identity] = name
            tensor = self.staging.get(name)
            if tensor is None or tensor.shape != shape or tensor.dtype != dtype:
                tensor = torch.empty(shape, dtype=dtype, device="cuda")
                self.staging[name] = tensor
            plan[name] = (tensor, nbytes, item)
        for name in list(self.staging):
            if name not in plan:
                del self.staging[name]
        mapped = time.perf_counter()
        stream = torch.cuda.current_stream()
        total = 0
        for tensor, nbytes, item in plan.values():
            if nbytes is None:
                continue  # Alias of a tensor copied above.
            _check(
                api.cuMemcpyDtoDAsync_v2(
                    tensor.data_ptr(),
                    self.opened[item["handle"]].value + item["offset"],
                    nbytes,
                    stream.cuda_stream,
                )
            )
            total += nbytes
        stream.synchronize()
        copied = time.perf_counter()
        if not cache:
            self._close_handles()
        return {name: tensor for name, (tensor, _, _) in plan.items()}, {
            "weight_map_seconds": mapped - started,
            "weight_peer_copy_seconds": copied - mapped,
            "weight_peer_bytes": total,
            "weight_handles_reopened": int(reopened),
        }

    def _close_handles(self):
        import torch

        if self.opened:
            torch.cuda.synchronize()
        for pointer in self.opened.values():
            _check(_driver().cuIpcCloseMemHandle(pointer))
        self.opened = {}
        self.payload = None

    def close(self):
        self._close_handles()
        self.staging = {}


def copy_exported_tensor(payload, name, destination):
    """One-off copy of a single exported tensor into an existing local tensor."""
    import torch

    item = payload["tensors"][name]
    if tuple(item["shape"]) != tuple(destination.shape) or getattr(
        torch, item["dtype"]
    ) != destination.dtype or not destination.is_contiguous():
        raise ValueError("exported tensor does not match its destination")
    local = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    if local == payload["device_uuid"]:
        with import_cuda_weights({**payload, "tensors": {name: item}}) as state:
            destination.copy_(state[name])
            torch.cuda.synchronize()
        return
    puller = CudaWeightPuller()
    try:
        state, _ = puller.pull({**payload, "tensors": {name: item}}, cache=False)
        destination.copy_(state[name])
        torch.cuda.synchronize()
    finally:
        puller.close()


def close_cuda_weight_cache(worker):
    cache = getattr(worker, "_paired_cuda_weight_cache", None)
    if cache is not None:
        cache.close()
        del worker._paired_cuda_weight_cache
    puller = getattr(worker, "_paired_cuda_weight_puller", None)
    if puller is not None:
        puller.close()
        del worker._paired_cuda_weight_puller


def load_cuda_weights(worker, payload, *, cache=False):
    import torch

    started = time.perf_counter()
    context = None
    local = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    if local != payload["device_uuid"]:
        # Peer GPU: pull into local staging, then run the stock loaders.
        with torch.inference_mode():
            if not hasattr(worker, "_paired_cuda_weight_puller"):
                worker._paired_cuda_weight_puller = CudaWeightPuller()
            state, timings = worker._paired_cuda_weight_puller.pull(
                payload, cache=cache
            )
            pulled = time.perf_counter()
            worker.model_runner.model.load_weights(state.items())
            torch.cuda.synchronize()
            timings["weight_load_seconds"] = time.perf_counter() - pulled
            timings["weight_total_seconds"] = time.perf_counter() - started
            return timings
    with torch.inference_mode():
        if cache:
            if not hasattr(worker, "_paired_cuda_weight_cache"):
                worker._paired_cuda_weight_cache = CudaWeightMappingCache()
            state, _ = worker._paired_cuda_weight_cache.get(payload)
        else:
            close_cuda_weight_cache(worker)
            context = import_cuda_weights(payload)
            state = context.__enter__()
        mapped = time.perf_counter()
        try:
            worker.model_runner.model.load_weights(state.items())
            torch.cuda.synchronize()
            loaded = time.perf_counter()
        finally:
            if context is not None:
                context.__exit__(None, None, None)
        finished = time.perf_counter()
    return {
        "weight_map_seconds": mapped - started,
        "weight_load_seconds": loaded - mapped,
        "weight_unmap_seconds": finished - loaded,
    }
