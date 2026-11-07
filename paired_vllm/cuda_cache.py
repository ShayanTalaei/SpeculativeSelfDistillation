"""Version-independent CUDA IPC target handoff; hidden bytes never enter CPU RAM.

PyTorch 2.9 cannot import torch 2.13's sharing-handle format. Use CUDA driver's
stable 64-byte IPC handles over explicitly owned cuMemAlloc slabs instead.
Producer leases stay alive until the consumer's peer copy has synchronized.
"""

import ctypes
from functools import lru_cache


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


@lru_cache(maxsize=1)
def _driver():
    api = ctypes.CDLL("libcuda.so.1")
    pointer = ctypes.c_uint64
    signatures = {
        "cuMemAlloc_v2": [ctypes.POINTER(pointer), ctypes.c_size_t],
        "cuMemFree_v2": [pointer],
        "cuMemcpyDtoDAsync_v2": [pointer, pointer, ctypes.c_size_t, ctypes.c_void_p],
        "cuIpcGetMemHandle": [ctypes.POINTER(_Handle), pointer],
        "cuIpcOpenMemHandle": [ctypes.POINTER(pointer), _Handle, ctypes.c_uint],
        "cuIpcCloseMemHandle": [pointer],
        "cuCtxGetCurrent": [ctypes.POINTER(ctypes.c_void_p)],
        "cuMemcpyPeerAsync": [
            pointer,
            ctypes.c_void_p,
            pointer,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ],
        "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
    }
    for name, args in signatures.items():
        function = getattr(api, name)
        function.argtypes = args
        function.restype = ctypes.c_int
    return api


def _check(status):
    if status:
        name = ctypes.c_char_p()
        _driver().cuGetErrorName(status, ctypes.byref(name))
        raise RuntimeError(f"CUDA target handoff failed: {name.value!r} ({status})")


class CudaTargetSlab:
    """An exporter-owned allocation outside torch/vLLM sleep allocators."""

    def __init__(self, hidden):
        import torch

        if not hidden.is_cuda or hidden.ndim != 2 or not hidden.is_contiguous():
            raise ValueError("target slab requires a contiguous CUDA matrix")
        self.device = hidden.device
        self.shape = tuple(hidden.shape)
        self.dtype = str(hidden.dtype).removeprefix("torch.")
        self.element_size = hidden.element_size()
        self.nbytes = hidden.numel() * self.element_size
        self.pointer = ctypes.c_uint64()
        self.closed = False
        self.uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        api = _driver()
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream()
            _check(api.cuMemAlloc_v2(ctypes.byref(self.pointer), self.nbytes))
            try:
                _check(
                    api.cuMemcpyDtoDAsync_v2(
                        self.pointer, hidden.data_ptr(), self.nbytes, stream.cuda_stream
                    )
                )
                stream.synchronize()
                handle = _Handle()
                _check(api.cuIpcGetMemHandle(ctypes.byref(handle), self.pointer))
                self.handle = bytes(handle)
            except BaseException:
                api.cuMemFree_v2(self.pointer)
                self.closed = True
                raise

    def descriptor(self, *, start, version, token_ids, lease):
        if self.closed or start < 0 or start + len(token_ids) > self.shape[0]:
            raise ValueError("invalid target slab slice")
        return {
            "format": "cuda-driver-hidden-v1",
            "handle": self.handle,
            "device_uuid": self.uuid,
            "dtype": self.dtype,
            "width": self.shape[1],
            "start": start,
            "allocation_bytes": self.nbytes,
            "version": str(version),
            "token_ids": list(token_ids),
            "lease": lease,
        }

    def close(self):
        import torch

        if not self.closed:
            with torch.cuda.device(self.device):
                _check(_driver().cuMemFree_v2(self.pointer))
            self.closed = True


def copy_cuda_targets(payloads, *, device, version, token_ids):
    """One direct peer copy per admission group into trainer-owned storage."""
    import torch

    if set(payloads) != set(token_ids):
        raise ValueError("teacher target request IDs differ from completions")
    destination = torch.device(device)
    if destination.type != "cuda":
        raise ValueError("resident targets require a CUDA training device")
    target_index = (
        destination.index
        if destination.index is not None
        else torch.cuda.current_device()
    )
    groups = {}
    for rid, payload in payloads.items():
        if payload["format"] != "cuda-driver-hidden-v1":
            raise ValueError("unknown CUDA target format")
        if payload["version"] != str(version) or payload["token_ids"] != list(
            token_ids[rid]
        ):
            raise ValueError("teacher cache version or token alignment mismatch")
        groups.setdefault(payload["lease"], []).append((rid, payload))
    result, api = {}, _driver()
    for rows in groups.values():
        rows.sort(key=lambda item: item[1]["start"])
        first = rows[0][1]
        source_index = next(
            (
                i
                for i in range(torch.cuda.device_count())
                if str(torch.cuda.get_device_properties(i).uuid) == first["device_uuid"]
            ),
            None,
        )
        if source_index is None:
            raise ValueError("teacher export GPU is not visible to the trainer")
        if source_index != target_index and not torch.cuda.can_device_access_peer(
            target_index, source_index
        ):
            raise ValueError("GPU-resident target handoff requires peer access")
        total = 0
        for rid, payload in rows:
            if any(
                payload[k] != first[k]
                for k in ("handle", "device_uuid", "dtype", "width", "allocation_bytes")
            ):
                raise ValueError("inconsistent target slab metadata")
            if payload["start"] != first["start"] + total:
                raise ValueError("target rows are not contiguous")
            total += len(token_ids[rid])
        if first["dtype"] not in ("bfloat16", "float16", "float32"):
            raise ValueError("unsupported target dtype")
        owned = torch.empty(
            (total, first["width"]),
            dtype=getattr(torch, first["dtype"]),
            device=torch.device("cuda", target_index),
        )
        offset = first["start"] * first["width"] * owned.element_size()
        nbytes = owned.numel() * owned.element_size()
        if (
            offset < 0
            or offset + nbytes > first["allocation_bytes"]
            or len(first["handle"]) != 64
        ):
            raise ValueError("target slice exceeds its allocation")
        source_pointer, source_context = ctypes.c_uint64(), ctypes.c_void_p()
        with torch.cuda.device(source_index):
            torch.cuda.current_stream()  # Ensure this device's primary context exists.
            _check(api.cuCtxGetCurrent(ctypes.byref(source_context)))
            _check(
                api.cuIpcOpenMemHandle(
                    ctypes.byref(source_pointer),
                    _Handle.from_buffer_copy(first["handle"]),
                    1,
                )
            )
        try:
            with torch.cuda.device(target_index):
                stream = torch.cuda.current_stream()
                target_context = ctypes.c_void_p()
                _check(api.cuCtxGetCurrent(ctypes.byref(target_context)))
                _check(
                    api.cuMemcpyPeerAsync(
                        owned.data_ptr(),
                        target_context,
                        source_pointer.value + offset,
                        source_context,
                        nbytes,
                        stream.cuda_stream,
                    )
                )
                stream.synchronize()
        finally:
            with torch.cuda.device(source_index):
                _check(api.cuIpcCloseMemHandle(source_pointer))
        start = 0
        for rid, _ in rows:
            count = len(token_ids[rid])
            result[rid] = owned[start : start + count]
            start += count
    return result


def copy_cuda_target(payload, *, device, version, token_ids):
    return copy_cuda_targets(
        {"row": payload}, device=device, version=version, token_ids={"row": token_ids}
    )["row"]
