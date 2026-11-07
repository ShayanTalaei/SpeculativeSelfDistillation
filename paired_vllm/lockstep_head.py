"""Device-only decision and definitive-token history update for lockstep decode."""

from paired_vllm.async_head import sample_with_draws


def jsd_scores(teacher, student):
    """Full-vocabulary FP32 JSD; compilation fuses the elementwise reductions."""
    import torch

    t, s = teacher.log_softmax(-1), student.log_softmax(-1)
    mixture = torch.logaddexp(t, s) - 0.6931471805599453
    return ((t.exp() * (t - mixture)).sum(-1) + (s.exp() * (s - mixture)).sum(-1)) / 2


def reverse_kl_scores(teacher, student):
    """Full-vocabulary KL(student || teacher), before sampling filters."""
    t, s = teacher.float().log_softmax(-1), student.float().log_softmax(-1)
    return (s.exp() * (s - t)).sum(-1)


def forward_kl_scores(teacher, student):
    """Full-vocabulary KL(teacher || student), before sampling filters."""
    t, s = teacher.float().log_softmax(-1), student.float().log_softmax(-1)
    return (t.exp() * (t - s)).sum(-1)


def routing_score(policy):
    return {
        "jsd": jsd_scores,
        "kl_reverse": reverse_kl_scores,
        "kl_forward": forward_kl_scores,
        "student": jsd_scores,  # Unused for unconditional routes.
        "teacher": jsd_scores,
    }[policy.partition(":")[0]]


class LockstepHead:
    def __init__(self, worker):
        self.worker = w = worker
        self.torch = torch = w.torch
        self.score = routing_score(w.settings.policy)
        if w.role == 1 and w.async_config.fused_jsd:
            # Keep the existing configuration name for compatibility; compile
            # the selected divergence, including reverse KL for ScienceQA.
            self.score = torch.compile(
                self.score,
                fullgraph=True,
                dynamic=True,
                options={"triton.cudagraphs": False},
            )
        self.buffers = {}
        self.host = [
            torch.empty(
                (3, w.settings.scheduler.max_requests),
                dtype=torch.int64,
                pin_memory=True,
            )
            for _ in range(2)
        ]
        self.host_index = 0
        self.has_projection = w.role == 1 or w.async_config.transfer_logits
        pool = torch.cuda.graph_pool_handle()
        for n in range(1, w.settings.scheduler.max_requests + 1):
            b = {
                "hidden": torch.zeros(
                    (n, w.config.model_config.hf_config.hidden_size),
                    device="cuda",
                    dtype=w.config.model_config.dtype,
                ),
                "teacher": torch.zeros(
                    (
                        n,
                        w.metadata["vocab"]
                        if w.async_config.transfer_logits
                        else w.teacher_width,
                    ),
                    device="cuda",
                    dtype=torch.float32
                    if w.async_config.transfer_logits
                    else w.config.model_config.dtype,
                ),
                "metadata": torch.zeros((3, n), device="cuda", dtype=torch.int64),
                "tokens": torch.zeros(n, device="cuda", dtype=torch.int64),
                "ones": torch.ones(n, device="cuda", dtype=torch.int32),
                "zeros": torch.zeros(n, device="cuda", dtype=torch.int32),
            }
            if self.has_projection:
                b["logits"] = torch.zeros(
                    (n, w.metadata["vocab"]), device="cuda", dtype=torch.float32
                )
            b["metadata"][0].fill_(-1)
            self.buffers[n] = b
            if w.settings.graphs and w.settings.head_graphs:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        if self.has_projection:
                            self.projection(b)
                        self.compute(b)
                torch.cuda.current_stream().wait_stream(stream)
                if self.has_projection:
                    projection_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(projection_graph, pool=pool):
                        self.projection(b)
                    b["projection_graph"] = projection_graph
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self.compute(b)
                b["graph"] = graph

    def projection(self, b):
        b["logits"].copy_(self.worker.runner.model.compute_logits(b["hidden"]))

    def project(self, b):
        if not self.has_projection:
            return
        if "projection_graph" in b:
            b["projection_graph"].replay()
        else:
            self.projection(b)

    def compute(self, b):
        w, torch = self.worker, self.torch
        native, slots, positions = b["metadata"].unbind(0)
        if w.role == 1:
            student = b["logits"]
            teacher = (
                b["teacher"]
                if w.async_config.transfer_logits
                else torch.nn.functional.linear(
                    b["teacher"], w.teacher_projection
                ).float()
            )
            if w.settings.policy == "student":
                route = torch.zeros(
                    student.shape[0], device=student.device, dtype=torch.bool
                )
            elif w.settings.policy == "teacher":
                route = torch.ones(
                    student.shape[0], device=student.device, dtype=torch.bool
                )
            else:
                scores = self.score(teacher, student)
                route = scores > w.async_config.threshold
            chosen = sample_with_draws(
                torch,
                torch.where(route[:, None], teacher, student),
                w.uniforms[slots, positions],
                w.settings.temperature,
                w.settings.top_k,
            )
            b["tokens"].copy_(chosen)
            w.output_sources[slots, positions] = route
        elif w.target_buffer is not None:
            w.target_buffer[slots, positions] = b["hidden"]
        w.output_tokens[slots, positions] = b["tokens"]
        w.runner.postprocess_sampled(
            native.to(torch.int32),
            b["tokens"][:, None],
            b["ones"],
            b["zeros"],
            query_start_loc=None,
        )

    def prepare(self, hidden, native, slots, positions):
        n = len(native)
        b = self.buffers[n]
        host = self.host[self.host_index][:, :n]
        self.host_index = (self.host_index + 1) % 2
        host.numpy()[:] = (native, slots, positions)
        b["metadata"].copy_(host, non_blocking=True)
        b["hidden"].copy_(hidden)
        return b

    def run(self, b):
        if "graph" in b:
            b["graph"].replay()
        else:
            self.compute(b)
