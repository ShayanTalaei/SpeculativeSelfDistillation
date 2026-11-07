"""Persistent student projection/sampling buffers for asynchronous rollouts."""

from __future__ import annotations


def sample_with_draws(torch, logits, draws, temperature, top_k):
    """Same inverse-CDF arithmetic as the reference keyed CPU sampler."""
    if temperature == 0:
        return logits.argmax(-1)
    values, indices = (
        logits.topk(min(top_k, logits.shape[-1]), dim=-1) if top_k else (logits, None)
    )
    cumulative = (values.float() / temperature).softmax(-1).cumsum(-1)
    selected = (cumulative.double() < draws[:, None]).sum(-1)
    selected.clamp_(max=values.shape[-1] - 1)
    return selected if indices is None else indices.gather(1, selected[:, None])[:, 0]


class DraftHead:
    """Exact-size graphs include sampling and the native device history update.

    Capture uses masked native request indices, before real requests exist.
    Host metadata has two owned slots; the caller retires GPU readers before
    reusing either slot, including MRV2's own two-slot UVA input buffers.
    """

    def __init__(self, worker):
        self.worker = worker
        self.torch = torch = worker.torch
        self.buffers = {}
        self.host = [
            torch.empty(
                (3, worker.settings.scheduler.max_requests),
                dtype=torch.int64,
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self.host_index = 0
        for n in range(1, worker.settings.scheduler.max_requests + 1):
            b = {
                "hidden": torch.zeros(
                    (n, worker.width),
                    device="cuda",
                    dtype=worker.config.model_config.dtype,
                ),
                "metadata": torch.zeros((3, n), device="cuda", dtype=torch.int64),
                "ones": torch.ones(n, device="cuda", dtype=torch.int32),
                "zeros": torch.zeros(n, device="cuda", dtype=torch.int32),
            }
            b["metadata"][0].fill_(-1)
            self.buffers[n] = b
            if worker.settings.graphs and worker.settings.head_graphs:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.compute(b)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.compute(b)
                b["graph"] = graph

    def compute(self, b):
        w, torch = self.worker, self.torch
        native, slots, positions = b["metadata"].unbind(0)
        hidden = b["hidden"]
        w.features[slots, positions] = hidden
        logits = w.runner.model.compute_logits(hidden).float()
        tokens = sample_with_draws(
            torch,
            logits,
            w.uniforms[slots, positions],
            w.settings.temperature,
            w.settings.top_k,
        )
        w.draft_tokens[slots, positions] = tokens
        w.runner.postprocess_sampled(
            native.to(torch.int32),
            tokens[:, None],
            b["ones"],
            b["zeros"],
            query_start_loc=None,
        )

    def __call__(self, hidden, native, slots, positions):
        n = len(native)
        b = self.buffers[n]
        host = self.host[self.host_index][:, :n]
        self.host_index = (self.host_index + 1) % len(self.host)
        host.numpy()[:] = (native, slots, positions)
        b["metadata"].copy_(host, non_blocking=True)
        b["hidden"].copy_(hidden)
        if "graph" in b:
            b["graph"].replay()
        else:
            self.compute(b)


class VerifyHead:
    """Bucketed graphs for exact full-vocabulary JSD and teacher sampling."""

    def __init__(self, worker):
        self.worker = w = worker
        self.torch = torch = w.torch
        self.buffers = {}
        maximum = w.settings.scheduler.max_requests * w.async_config.chunk_size
        self.host = torch.empty((3, maximum), dtype=torch.int64, pin_memory=True)
        sizes = sorted({maximum, *(1 << i for i in range(maximum.bit_length()))})
        pool = torch.cuda.graph_pool_handle()
        for n in sizes:
            b = {
                "teacher": torch.zeros(
                    (n, w.config.model_config.hf_config.hidden_size),
                    device="cuda",
                    dtype=w.config.model_config.dtype,
                ),
                "student": torch.zeros(
                    (n, w.width), device="cuda", dtype=w.config.model_config.dtype
                ),
                "metadata": torch.zeros((3, n), device="cuda", dtype=torch.int64),
            }
            self.buffers[n] = b
            if w.settings.graphs and w.settings.head_graphs:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.compute(b)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    b["values"] = self.compute(b)
                b["graph"] = graph

    def compute(self, b):
        w, torch = self.worker, self.torch
        teacher = w.runner.model.compute_logits(b["teacher"]).float()
        student = torch.nn.functional.linear(b["student"], w.student_projection).float()
        t, s = teacher.log_softmax(-1), student.log_softmax(-1)
        mixture = torch.logaddexp(t, s) - 0.6931471805599453
        scores = (
            (t.exp() * (t - mixture)).sum(-1) + (s.exp() * (s - mixture)).sum(-1)
        ) / 2
        route = scores > w.async_config.threshold
        original, slots, positions = b["metadata"].unbind(0)
        candidates = sample_with_draws(
            torch,
            teacher,
            w.uniforms[slots, positions],
            w.settings.temperature,
            w.settings.top_k,
        )
        chosen = torch.where(route, candidates, original)
        return torch.stack((chosen.double(), route.double(), scores.double()), -1)

    def __call__(self, teacher, student, drafts):
        n = teacher.shape[0]
        size = next(size for size in self.buffers if size >= n)
        b = self.buffers[size]
        w = self.worker
        self.host.numpy()[:, :n] = (
            [t for d in drafts for t in d.tokens],
            [w.slots[d.request_id] for d in drafts for _ in d.tokens],
            [d.start + i for d in drafts for i in range(len(d.tokens))],
        )
        b["metadata"][:, :n].copy_(self.host[:, :n], non_blocking=True)
        b["teacher"][:n].copy_(teacher)
        b["student"][:n].copy_(student)
        if "graph" in b:
            b["graph"].replay()
            return b["values"][:n].tolist()
        return self.compute(b)[:n].tolist()
