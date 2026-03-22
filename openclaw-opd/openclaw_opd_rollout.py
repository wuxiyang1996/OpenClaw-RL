import asyncio
import atexit
import os
import queue
import threading
import time

from openclaw_opd_api_server import OpenClawOPDAPIServer
from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import eval_rollout
from slime.utils.async_utils import run
from slime.utils.types import Sample

_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer):
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            _global_worker = AsyncRolloutWorker(args, data_buffer)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


def _parse_adapter_ports() -> list[tuple[str, str]]:
    """Return ``[(adapter_name, port), ...]`` from environment variables.

    When ``PORT_A`` / ``PORT_B`` are set, we are in federated (dual-server)
    mode.  The adapter names default to ``lora_a`` / ``lora_b`` but can be
    overridden with ``ADAPTER_NAME_A`` / ``ADAPTER_NAME_B``.

    Falls back to a single ``(None, PORT)`` entry for non-federated mode.
    """
    port_a = os.getenv("PORT_A")
    port_b = os.getenv("PORT_B")
    if port_a and port_b:
        name_a = os.getenv("ADAPTER_NAME_A", "lora_a")
        name_b = os.getenv("ADAPTER_NAME_B", "lora_b")
        return [(name_a, port_a), (name_b, port_b)]
    return [(None, os.getenv("PORT", "30000"))]


class AsyncRolloutWorker:
    """Manages one or more ``OpenClawOPDAPIServer`` instances.

    In single-server mode (default), behaviour is identical to the original.
    In federated mode (``PORT_A`` and ``PORT_B`` set), two servers are started
    on different ports, each with its own output queue and adapter name.
    """

    def __init__(self, args, data_buffer):
        self.args = args
        self.data_buffer = data_buffer
        self.running = True
        self.worker_thread = None
        self._submission_enabled = threading.Event()

        adapter_ports = _parse_adapter_ports()
        self._is_federated = len(adapter_ports) > 1

        self._servers: list[OpenClawOPDAPIServer] = []
        self._queues: dict[str | None, queue.Queue] = {}
        self.output_queue = queue.Queue(maxsize=100000)

        if self._is_federated:
            for adapter_name, port in adapter_ports:
                q = queue.Queue(maxsize=100000)
                self._queues[adapter_name] = q
                server = OpenClawOPDAPIServer(
                    args=args,
                    output_queue=q,
                    submission_enabled=self._submission_enabled,
                    adapter_name=adapter_name,
                )
                server.port = int(port)
                self._servers.append(server)
        else:
            self._queues[None] = self.output_queue
            server = OpenClawOPDAPIServer(
                args=args,
                output_queue=self.output_queue,
                submission_enabled=self._submission_enabled,
            )
            self._servers.append(server)

    async def continuous_worker_loop(self):
        while self.running:
            await asyncio.sleep(1.0)

    def worker_thread_func(self):
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        for server in self._servers:
            server.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()

    def stop(self):
        self.running = False
        self._submission_enabled.clear()
        for server in self._servers:
            server.stop()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def pause_submission(self):
        if self._submission_enabled.is_set():
            self._submission_enabled.clear()
            for server in self._servers:
                server.purge_record_files()
            print("[OpenClawOPDWorker] submission paused")

    def resume_submission(self):
        if not self._submission_enabled.is_set():
            self._submission_enabled.set()
            print("[OpenClawOPDWorker] submission resumed")

    def get_completed_groups(self, adapter_name: str | None = None) -> list[tuple]:
        q = self._queues.get(adapter_name, self.output_queue)
        completed = []
        while True:
            try:
                completed.append(q.get_nowait())
            except queue.Empty:
                break
        return completed

    def get_all_completed_groups(self) -> dict[str | None, list[tuple]]:
        return {name: self.get_completed_groups(name) for name in self._queues}

    def get_queue_size(self, adapter_name: str | None = None) -> int:
        q = self._queues.get(adapter_name, self.output_queue)
        return q.qsize()

    def get_total_queue_size(self) -> int:
        return sum(q.qsize() for q in self._queues.values())

    def drain_all_eval_scores(self) -> list[float]:
        scores = []
        for server in self._servers:
            scores.extend(server.drain_eval_scores())
        return scores

    def reset_all_eval_scores(self):
        for server in self._servers:
            server.reset_eval_scores()


async def _drain_output_queue(args, worker: AsyncRolloutWorker) -> list[list[Sample]]:
    target_data_size = args.rollout_batch_size
    data: list[list[Sample]] = []
    completed_groups: dict[int, list[Sample]] = {}
    start = time.time()
    last_progress = start

    while len(data) < target_data_size:
        completed = worker.get_completed_groups()
        if completed:
            last_progress = time.time()
            for group_id, group in completed:
                completed_groups[group_id] = group

        for group_id in list(completed_groups.keys()):
            if len(data) >= target_data_size:
                break
            group = completed_groups.pop(group_id)
            if any(sample.status == Sample.Status.ABORTED for sample in group):
                continue
            data.append(group)

        if time.time() - last_progress > 30:
            print(
                f"[OpenClawOPDWorker] waiting for valid OPD samples: {len(data)}/{target_data_size}, "
                f"queue={worker.get_queue_size()}",
                flush=True,
            )
            last_progress = time.time()

        if len(data) < target_data_size:
            await asyncio.sleep(0.05)

    data.sort(key=lambda group: group[0].index if group and group[0].index is not None else -1)
    print(f"[OpenClawOPDWorker] drained {len(data)} groups in {time.time() - start:.2f}s", flush=True)
    return data


async def _drain_federated_queues(
    args,
    worker: AsyncRolloutWorker,
) -> dict[str, list[list[Sample]]]:
    """Drain all per-adapter queues until each has at least ``rollout_batch_size`` samples."""
    target = args.rollout_batch_size
    result: dict[str, list[list[Sample]]] = {name: [] for name in worker._queues if name is not None}
    pending: dict[str, dict[int, list[Sample]]] = {name: {} for name in result}
    start = time.time()
    last_progress = start

    while any(len(v) < target for v in result.values()):
        all_groups = worker.get_all_completed_groups()
        got_any = False
        for adapter_name, groups in all_groups.items():
            if adapter_name is None:
                continue
            for group_id, group in groups:
                pending[adapter_name][group_id] = group
                got_any = True

        if got_any:
            last_progress = time.time()

        for adapter_name in list(result.keys()):
            for gid in list(pending[adapter_name].keys()):
                if len(result[adapter_name]) >= target:
                    break
                group = pending[adapter_name].pop(gid)
                if any(s.status == Sample.Status.ABORTED for s in group):
                    continue
                result[adapter_name].append(group)

        if time.time() - last_progress > 30:
            counts = {k: len(v) for k, v in result.items()}
            sizes = {k: worker.get_queue_size(k) for k in result}
            print(
                f"[OpenClawOPDWorker] federated drain: {counts}/{target}, queues={sizes}",
                flush=True,
            )
            last_progress = time.time()

        if any(len(v) < target for v in result.values()):
            await asyncio.sleep(0.05)

    for name, data in result.items():
        data.sort(key=lambda g: g[0].index if g and g[0].index is not None else -1)

    elapsed = time.time() - start
    counts = {k: len(v) for k, v in result.items()}
    print(f"[OpenClawOPDWorker] federated drain done: {counts} in {elapsed:.2f}s", flush=True)
    return result


def generate_rollout_openclaw_opd(args, rollout_id, data_buffer, evaluation=False):
    worker = get_global_worker(args, data_buffer)

    if evaluation:
        eval_output, _ = run(eval_rollout(args, rollout_id))
        return eval_output

    worker.reset_all_eval_scores()
    worker.resume_submission()

    if worker._is_federated:
        per_adapter = run(_drain_federated_queues(args, worker))
        worker.pause_submission()

        all_samples: list[list[Sample]] = []
        for samples in per_adapter.values():
            all_samples.extend(samples)

        extra_metrics = None
        eval_scores = worker.drain_all_eval_scores()
        if eval_scores:
            extra_metrics = {"rollout/prm_eval_score": sum(eval_scores) / len(eval_scores)}

        return RolloutFnTrainOutput(samples=all_samples, metrics=extra_metrics)
    else:
        completed_samples = run(_drain_output_queue(args, worker))
        worker.pause_submission()

        extra_metrics = None
        eval_scores = worker.drain_all_eval_scores()
        if eval_scores:
            extra_metrics = {"rollout/prm_eval_score": sum(eval_scores) / len(eval_scores)}
            print(
                f"[OpenClawOPDWorker] prm_eval_score={extra_metrics['rollout/prm_eval_score']:.4f} "
                f"(n={len(eval_scores)})",
                flush=True,
            )

        return RolloutFnTrainOutput(samples=completed_samples, metrics=extra_metrics)


atexit.register(stop_global_worker)
