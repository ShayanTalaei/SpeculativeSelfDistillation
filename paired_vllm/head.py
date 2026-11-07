"""Student-side teacher projection, routing, and selected-distribution sampling.

The caller must wait for receive completion before evaluating this head.
"""


class DecisionHead:
    def __init__(
        self,
        torch,
        weight,
        features,
        route,
        candidates,
        policy,
        temperature=0.0,
        top_k=0,
        enable_graphs=False,
    ):
        self.torch, self.weight, self.features = torch, weight, features
        self.route, self.candidates, self.policy = route, candidates, policy
        self.temperature, self.top_k = temperature, top_k
        self.enable_graphs = enable_graphs
        self.graphs = {}
        self.static_own = None

    def graphed(self, n, own):
        torch = self.torch
        padded = min(len(self.features), 1 << (n - 1).bit_length())
        if self.static_own is None:
            self.static_own = torch.zeros(
                (len(self.features), own.shape[-1]), device=own.device, dtype=own.dtype
            )
        self.static_own[:n].copy_(own)
        self.static_own[n:padded].zero_()
        self.features[n:padded].zero_()
        self.route[n:padded].zero_()
        key = (self.policy, self.temperature, self.top_k, padded)
        if key not in self.graphs:
            # Capture projection, routing, and sampling together. Stable buffers
            # survive weight refreshes; no token decision crosses a transaction.
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.compute(padded, self.static_own[:padded])
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.compute(padded, self.static_own[:padded])
            self.graphs[key] = (graph, output)
        graph, output = self.graphs[key]
        graph.replay()
        return output[:n]

    def compute(self, n, own):
        torch = self.torch
        teacher_logits = torch.nn.functional.linear(
            self.features[:n], self.weight
        ).float()
        if self.policy == "teacher":
            self.route[:n].fill_(1)
        elif self.policy == "student":
            self.route[:n].zero_()
        elif self.policy != "alternating":
            teacher, student = teacher_logits.log_softmax(-1), own.log_softmax(-1)
            if self.policy.startswith("kl_forward:"):
                score = (teacher.exp() * (teacher - student)).sum(-1)
            elif self.policy.startswith("kl_reverse:"):
                score = (student.exp() * (student - teacher)).sum(-1)
            elif self.policy.startswith("jsd:"):
                mixture = self.torch.logaddexp(teacher, student) - 0.6931471805599453
                score = (
                    (teacher.exp() * (teacher - mixture)).sum(-1)
                    + (student.exp() * (student - mixture)).sum(-1)
                ) / 2
            else:
                raise ValueError(f"unsupported policy {self.policy}")
            self.route[:n].copy_((score > float(self.policy.split(":")[1])).int())
        selected_logits = torch.where(self.route[:n, None].bool(), teacher_logits, own)
        if self.temperature == 0:
            self.candidates[:n].copy_(selected_logits.argmax(-1))
        else:
            if self.top_k > 0:
                values, indices = selected_logits.topk(
                    min(self.top_k, selected_logits.shape[-1]), dim=-1
                )
                chosen = torch.multinomial((values / self.temperature).softmax(-1), 1)
                self.candidates[:n].copy_(indices.gather(-1, chosen).squeeze(-1))
            else:
                self.candidates[:n].copy_(
                    torch.multinomial(
                        (selected_logits / self.temperature).softmax(-1), 1
                    ).squeeze(-1)
                )
        return teacher_logits

    def __call__(self, own, output_indices):
        n = len(output_indices)
        if self.policy == "alternating":
            self.route[:n].copy_(
                self.torch.tensor(
                    [int(k % 2 == 0) for k in output_indices],
                    device=self.route.device,
                    dtype=self.route.dtype,
                )
            )
        if self.enable_graphs and n:
            return self.graphed(n, own)
        return self.compute(n, own)
