"""Stock continuous scheduling on the existing student worker and KV cache."""

import time


def evaluate_student(adapter, payload):
    from vllm import SamplingParams
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.outputs import AsyncModelRunnerOutput
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    if adapter.role != 1:
        raise ValueError("student evaluation cannot execute on the teacher")
    prompts = payload["student"]
    if any(
        not ids or len(ids) + payload["max_tokens"] > adapter.settings.max_model_len
        for ids in prompts
    ):
        raise ValueError("evaluation prompt exceeds the configured context length")

    def clear_requests():
        output = SchedulerOutput.make_empty()
        output.finished_req_ids = set(adapter.runner.req_states.req_id_to_index)
        adapter.worker.execute_model(output)
        adapter.runner.execute_model_state = None

    # Training and evaluation reuse allocations, but never request/KV metadata.
    clear_requests()
    adapter.requests.clear()
    scheduler = Scheduler(
        adapter.config,
        adapter.kv_cache_config,
        StructuredOutputManager(adapter.config),
        block_size=adapter.settings.scheduler.block_size,
    )
    requests = []
    for i, ids in enumerate(prompts):
        params = SamplingParams(
            temperature=payload["temperature"],
            top_k=payload["top_k"] or -1,
            max_tokens=payload["max_tokens"],
            stop_token_ids=list(payload["stop_tokens"]),
            detokenize=False,
        )
        request = Request(f"eval-{i}", ids, params, pooling_params=None)
        requests.append(request)
        scheduler.add_request(request)
    started = time.perf_counter()
    steps = max_batch = refills = 0
    try:
        # Queue the whole group so a completed answer immediately frees a slot
        # for another question instead of waiting for its original batch.
        while scheduler.has_requests():
            scheduled = scheduler.schedule()
            batch = len(scheduled.num_scheduled_tokens)
            max_batch = max(max_batch, batch)
            if steps and scheduled.scheduled_new_reqs:
                refills += len(scheduled.scheduled_new_reqs)
            result = adapter.worker.execute_model(scheduled)
            if result is None:
                result = adapter.worker.sample_tokens(
                    scheduler.get_grammar_bitmask(scheduled)
                )
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            scheduler.update_from_output(scheduled, result)
            steps += 1
    finally:
        clear_requests()
    seconds = time.perf_counter() - started
    rows = [
        {
            "token_ids": list(r.output_token_ids),
            "provenance": ["student"] * len(r.output_token_ids),
        }
        for r in requests
    ]
    if rows:
        rows[0]["timings"] = {
            "generation_seconds": seconds,
            "teacher_forward_steps": 0,
            "scheduler_steps": steps,
            "max_batch": max_batch,
            "continuous_refills": refills,
            "completion_tokens": sum(len(r["token_ids"]) for r in rows),
        }
    return rows
