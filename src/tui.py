import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

import os
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Static, ProgressBar
from textual.worker import get_current_worker


import GPUalgorithm as cuda_alignment
import offtarget


CHUNK_DIR = "data/chunks"
CHUNK_FILENAMES = [f"chunk_{i:06d}.npy" for i in range(1, 31)]

SAMPLE_SIZE = 500000

KERNEL_MODE = "warp"   # "warp" | "banded" | "tiled" | "diagonal"

GPU_POLL_SECONDS = 2.0   # how often to re-read GPU utilisation while running

GUIDE_RNA = "GACCCCCTCCACCCCGCCTC"
MAX_MISMATCHES = 4
SCAN_BOTH_STRANDS = True
TOP_OFF_TARGETS = 5


class GeneWeaverApp(App):

    CSS_PATH = "tui.tcss"

    TITLE = "GeneWeaver - Genome Processing Dashboard"

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_files = []
        self._dask_client = None
        self._dask_cluster = None
        self._pipeline_running = False

    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="main-panel"):
            yield Static("🧬 GENEWEAVER", classes="title")

            yield Static("ALIGNMENT DASHBOARD", classes="section-title")
            yield Static("Alignment Progress", id="alignment-progress-label")
            yield Static("Chunk Pair: 0 / 9", id="chunk-pair")

            yield ProgressBar(
                total=9,
                show_eta=False,
                id="alignment_progress",
            )

            yield Static("GPU STATUS", classes="section-title")
            yield Static("GPU: Not Connected", id="gpu-status")
            yield Static("GPU Count: --", id="gpu-count")
            yield Static("Devices: --", id="gpu-devices")
            yield Static("GPU Memory: --", id="gpu-memory")
            yield Static("GPU Utilization: --", id="gpu-utilization")
            yield Static("CUDA Status: Not Available", id="cuda-status")
            yield Static("Kernel: --", id="kernel-info")
            yield Static("Occupancy: --", id="occupancy-info")

            yield Static("DASK CLUSTER", classes="section-title")
            yield Static("Cluster: Not started", id="dask-status")
            yield Static("Worker -> GPU: --", id="dask-pinning")
            yield Static("Pair distribution: --", id="dask-distribution")

            yield Static("RESULTS", classes="section-title")

            yield Static("GPU Alignment", id="gpu-result-title")
            yield Static("Chunk Length: --", id="chunk-length")
            yield Static("VRAM / pair: --", id="vram-estimate")
            yield Static("Average Time: --", id="gpu-average-time")
            yield Static("Throughput: --", id="gpu-throughput")
            yield Static("Total Cells: --", id="gpu-total-cells")


            yield Static("OFF-TARGET SEVERITY", classes="section-title")
            yield Static("Guide: --", id="guide-info")
            yield Static("Scan: not started", id="offtarget-status")
            yield Static("", id="offtarget-ranking")

            yield Static("Genome Chunking Progress (Dask)", classes="section-title")

            yield ProgressBar(
                total=10,
                show_eta=False,
                id="chunk_progress",
            )

            yield Static("Status: Ready", id="status")
            yield Static("Current file: None", id="current_file")
            yield Static("Chunks: 0 / 10", id="chunk_count")

        yield Footer()

    def on_mount(self) -> None:
        self._refresh_gpu_status()
        self._set_kernel_info(self._kernel_label())
        self._set_guide_info()

        candidate_paths = [Path(CHUNK_DIR) / fname for fname in CHUNK_FILENAMES]
        missing = [p for p in candidate_paths if not p.exists()]

        progress_bar = self.query_one("#chunk_progress", ProgressBar)
        progress_bar.update(total=len(candidate_paths))

        if missing:
            shown = ", ".join(p.name for p in missing[:3])
            more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
            self.query_one("#status", Static).update(
                f"Status: Missing {len(missing)}/{len(candidate_paths)} chunk file(s) "
                f"in '{CHUNK_DIR}': {shown}{more}"
            )
            return

        self.chunk_files = candidate_paths
        self.query_one("#status", Static).update("Status: Starting Dask cluster...")
        self._pipeline_running = True
        self.run_pipeline_worker()
        self.poll_gpu_utilization()

    def on_unmount(self) -> None:
        self._shutdown_dask()

    def _kernel_label(self) -> str:
        sq = f"{cuda_alignment.TILE_DIM}x{cuda_alignment.TILE_DIM} tile"
        wt = f"{cuda_alignment.TILE_ROWS}x{cuda_alignment.TILE_COLS} tile"
        return {
            "warp": f"warp-tile ({wt}, O(n+m) VRAM)",
            "banded": f"banded tiled shared-memory ({sq}, O(n+m) VRAM)",
            "tiled": f"tiled shared-memory ({sq}, full DP matrix)",
            "diagonal": "diagonal (global memory, full DP matrix)",
        }.get(KERNEL_MODE, KERNEL_MODE)

    def _set_guide_info(self) -> None:
        strands = "both strands" if SCAN_BOTH_STRANDS else "forward strand"
        self.query_one("#guide-info", Static).update(
            f"Guide: 5'-{GUIDE_RNA}-3'  ({len(GUIDE_RNA)} bp), PAM NGG/NAG, "
            f"<= {MAX_MISMATCHES} mismatches, {strands}"
        )

    def _set_offtarget_status(self, text: str) -> None:
        self.query_one("#offtarget-status", Static).update(f"Scan: {text}")

    def _set_offtarget_ranking(self, hits: list) -> None:
        self.query_one("#offtarget-ranking", Static).update(
            offtarget.render_ranking(hits, TOP_OFF_TARGETS)
        )

    def _set_occupancy(self, bp: int) -> None:
        try:
            occ = cuda_alignment.occupancy_report(bp, bp, KERNEL_MODE)
        except Exception:
            return
        self.query_one("#occupancy-info", Static).update(
            f"Occupancy: {occ['threads_per_block']} thr/block "
            f"({occ['warps_per_block']} warps), "
            f"{occ['shared_per_block_b']/1024:.1f} KB shared, "
            f"{occ['lane_efficiency']*100:.0f}% lanes, "
            f"{occ['launches_per_pair']:,} launches/pair"
        )

    def _set_utilization_only(self, devices) -> None:
        vals = [d["utilization_pct"] for d in devices if d.get("utilization_pct") is not None]
        if not vals:
            return
        detail = "  ".join(f"[{d['index']}] {d['utilization_pct']:.0f}%"
                           for d in devices if d.get("utilization_pct") is not None)
        avg = sum(vals) / len(vals)
        self.query_one("#gpu-utilization", Static).update(
            f"GPU Utilization: {avg:.0f}%" + (f"   {detail}" if len(vals) > 1 else "")
        )

    @work(thread=True, exclusive=False)
    def poll_gpu_utilization(self) -> None:
        poll_worker = get_current_worker()
        while not poll_worker.is_cancelled and self._pipeline_running:
            try:
                self.call_from_thread(
                    self._set_utilization_only, cuda_alignment.list_gpu_devices()
                )
            except Exception:
                pass
            time.sleep(GPU_POLL_SECONDS)

    def _refresh_gpu_status(self) -> None:
        try:
            info = cuda_alignment.gpu_status_info()
        except Exception as exc:  # never let device probing kill the dashboard
            self.query_one("#gpu-status", Static).update(f"GPU: probe failed ({exc})")
            return
        self._set_gpu_status_widgets(info)

    def _shutdown_dask(self) -> None:
        client, cluster = self._dask_client, self._dask_cluster
        self._dask_client, self._dask_cluster = None, None
        for obj in (client, cluster):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(f"Status: {text}")

    def _set_current_file(self, text: str) -> None:
        self.query_one("#current_file", Static).update(f"Current file: {text}")

    def _set_chunk_progress(self, done: int, total: int) -> None:
        self.query_one("#chunk_progress", ProgressBar).update(total=total, progress=done)
        self.query_one("#chunk_count", Static).update(f"Chunks: {done} / {total}")

    def _set_alignment_label(self, text: str) -> None:
        self.query_one("#alignment-progress-label", Static).update(text)

    def _set_alignment_progress(self, done: int, total: int) -> None:
        self.query_one("#alignment_progress", ProgressBar).update(total=total, progress=done)
        self.query_one("#chunk-pair", Static).update(f"Chunk Pair: {done} / {total}")

    def _set_dask_status(self, text: str) -> None:
        self.query_one("#dask-status", Static).update(f"Cluster: {text}")

    def _set_dask_pinning(self, text: str) -> None:
        self.query_one("#dask-pinning", Static).update(f"Worker -> GPU: {text}")

    def _set_dask_distribution(self, text: str) -> None:
        self.query_one("#dask-distribution", Static).update(f"Pair distribution: {text}")

    def _set_kernel_info(self, text: str) -> None:
        self.query_one("#kernel-info", Static).update(f"Kernel: {text}")

    def _set_gpu_status_widgets(self, info: dict) -> None:
        n_gpus = info.get("count", 0)
        devices = info.get("devices") or []

        if info["simulator"]:
            self.query_one("#gpu-status", Static).update("GPU: Simulator (no hardware detected)")
            self.query_one("#cuda-status", Static).update("CUDA Status: Simulator mode")
        elif info["available"]:
            name = info["device_name"] or "GPU"
            self.query_one("#gpu-status", Static).update(f"GPU: Connected ({name})")
            self.query_one("#cuda-status", Static).update("CUDA Status: Available")
        else:
            self.query_one("#gpu-status", Static).update("GPU: Not Connected")
            self.query_one("#cuda-status", Static).update("CUDA Status: Not Available")

        suffix = ""
        if info.get("forced_count"):
            suffix = " (forced via FORCE_GPU_COUNT)"
        elif info["simulator"]:
            suffix = " (simulated)"

        if n_gpus == 0:
            self.query_one("#gpu-count", Static).update(f"GPU Count: 0 - no CUDA device found{suffix}")
        else:
            label = "GPU" if n_gpus == 1 else "GPUs"
            self.query_one("#gpu-count", Static).update(f"GPU Count: {n_gpus} {label} detected{suffix}")

        if devices:
            lines = []
            for d in devices:
                mem = (f" - {d['memory_total_gb']:.1f} GB"
                       if d.get("memory_total_gb") is not None else "")
                lines.append(f"  [{d['index']}] {d['name']}{mem}")
            self.query_one("#gpu-devices", Static).update("Devices:\n" + "\n".join(lines))
        else:
            self.query_one("#gpu-devices", Static).update("Devices: none")

        if info["memory_free_gb"] is not None and info["memory_total_gb"] is not None:
            scope = " (all GPUs)" if n_gpus > 1 else ""
            self.query_one("#gpu-memory", Static).update(
                f"GPU Memory: {info['memory_free_gb']:.2f} / {info['memory_total_gb']:.2f} GB free{scope}"
            )
        else:
            self.query_one("#gpu-memory", Static).update("GPU Memory: N/A")

        if info["utilization_pct"] is not None:
            scope = " (avg)" if n_gpus > 1 else ""
            self.query_one("#gpu-utilization", Static).update(
                f"GPU Utilization: {info['utilization_pct']:.0f}%{scope}"
            )
        else:
            self.query_one("#gpu-utilization", Static).update("GPU Utilization: N/A")

    def _set_memory_info(self, bp: int, per_pair_bytes: int, free_bytes) -> None:
        self.query_one("#chunk-length", Static).update(f"Chunk Length: {bp:,} bp")
        free_txt = f", {free_bytes/1e9:.2f} GB free" if free_bytes else ""
        if per_pair_bytes >= 1e9:
            need = f"{per_pair_bytes/1e9:,.2f} GB"
        else:
            need = f"{per_pair_bytes/1e6:,.2f} MB"
        self.query_one("#vram-estimate", Static).update(f"VRAM / pair: {need}{free_txt}")

    def _set_gpu_results(self, avg_time_s: float, avg_throughput: float, total_cells: int) -> None:
        self.query_one("#gpu-average-time", Static).update(f"Average Time: {avg_time_s*1000:.2f} ms")
        self.query_one("#gpu-throughput", Static).update(f"Throughput: {avg_throughput:,.0f} cells/sec")
        self.query_one("#gpu-total-cells", Static).update(f"Total Cells: {total_cells:,}")


    @work(thread=True)
    def run_pipeline_worker(self) -> None:
        worker = get_current_worker()

        gpu_info = cuda_alignment.gpu_status_info()
        self.call_from_thread(self._set_gpu_status_widgets, gpu_info)

        n_gpus = max(1, gpu_info.get("count", 0))

        self.call_from_thread(self._set_status, f"Starting Dask cluster ({n_gpus} worker(s))...")
        try:
            client, cluster = cuda_alignment.create_dask_cluster(n_gpus)
        except Exception as exc:
            self.call_from_thread(self._set_dask_status, f"failed to start ({exc})")
            self.call_from_thread(
                self._set_status, "Dask cluster failed to start - genome parsing aborted"
            )
            self._pipeline_running = False
            return

        self._dask_client, self._dask_cluster = client, cluster
        n_workers = len(getattr(cluster, "workers", {})) or n_gpus
        self.call_from_thread(
            self._set_dask_status,
            f"LocalCluster up - {n_workers} worker process(es), 1 thread each",
        )

        pin_map = cuda_alignment.worker_gpu_map(client, n_gpus)
        if pin_map:
            self.call_from_thread(
                self._set_dask_pinning,
                ", ".join(f"{addr.rsplit('/', 1)[-1]} -> GPU {gid}" for addr, gid in pin_map),
            )
        else:
            self.call_from_thread(self._set_dask_pinning, "unpinned (worker list unavailable)")

        try:
            self.call_from_thread(self._set_status, "Parsing genome chunks on Dask workers...")

            def parse_progress_cb(done, total, filename):
                if worker.is_cancelled:
                    return
                self.call_from_thread(self._set_current_file, filename)
                self.call_from_thread(self._set_chunk_progress, done, total)
                self.call_from_thread(
                    self._set_status, f"Parsed chunk {done}/{total} (Dask)"
                )

            try:
                sequences = cuda_alignment.parse_paths_dask(
                    [str(p) for p in self.chunk_files],
                    SAMPLE_SIZE,
                    client=client,
                    progress_callback=parse_progress_cb,
                )
            except Exception as exc:
                self.call_from_thread(self._set_status, f"Dask parsing failed: {exc}")
                return

            if worker.is_cancelled:
                return

            self.call_from_thread(self._set_status, "Scanning for off-target sites...")
            self.call_from_thread(self._set_offtarget_status, "running on Dask workers...")
            try:
                def scan_progress_cb(done, total, label, found):
                    self.call_from_thread(
                        self._set_offtarget_status,
                        f"{done}/{total} chunks scanned, {found} hit(s) in {label}",
                    )

                off_targets = offtarget.scan_chunks_dask(
                    sequences,
                    [p.name for p in self.chunk_files],
                    GUIDE_RNA,
                    max_mismatches=MAX_MISMATCHES,
                    client=client,
                    both_strands=SCAN_BOTH_STRANDS,
                    progress_callback=scan_progress_cb,
                )
                by_level = {}
                for hit in off_targets:
                    by_level[hit["severity"]] = by_level.get(hit["severity"], 0) + 1
                summary = ", ".join(f"{by_level[k]} {k}" for k in ("HIGH", "MEDIUM", "LOW")
                                    if k in by_level) or "none"
                self.call_from_thread(
                    self._set_offtarget_status,
                    f"{len(off_targets):,} site(s) - {summary}",
                )
                self.call_from_thread(self._set_offtarget_ranking, off_targets)
            except Exception as exc:
                self.call_from_thread(self._set_offtarget_status, f"failed: {exc}")

            if worker.is_cancelled:
                return

            lengths = [cuda_alignment.sequence_length(s) for s in sequences]
            longest = max(lengths)
            fits, needed, free = cuda_alignment.check_device_capacity(
                longest, longest, KERNEL_MODE
            )
            self.call_from_thread(self._set_memory_info, longest, needed, free)
            self.call_from_thread(self._set_occupancy, longest)

            if not fits:
                banded = cuda_alignment.estimate_device_bytes(longest, longest, "banded")
                self.call_from_thread(
                    self._set_status,
                    f"Aborted: kernel '{KERNEL_MODE}' needs {needed/1e9:,.2f} GB of VRAM at "
                    f"{longest:,} bp. Set KERNEL_MODE = \"banded\" ({banded/1e6:.1f} MB) "
                    f"or lower SAMPLE_SIZE.",
                )
                return

            self.call_from_thread(
                self._set_status, f"Aligning on {n_gpus} GPU(s) via Dask..."
            )
            self.call_from_thread(self._set_alignment_label, "Alignment Progress (Dask)")
            self.call_from_thread(self._set_alignment_progress, 0, len(sequences) - 1)

            def gpu_progress_cb(done, total, pair_label):
                self.call_from_thread(self._set_alignment_progress, done, total)

            try:
                gpu_results, n_gpus_used = cuda_alignment.align_sequence_pairs_gpu_dask(
                    sequences,
                    client=client,
                    n_gpus=n_gpus,
                    progress_callback=gpu_progress_cb,
                    kernel=KERNEL_MODE,
                )
            except Exception as exc:
                self.call_from_thread(self._set_status, f"Dask GPU alignment failed: {exc}")
                return

            fallback = next(
                (r["shared_kernel_error"] for r in gpu_results if r.get("shared_kernel_error")),
                None,
            )
            kernels_used = sorted({r.get("kernel", "unknown") for r in gpu_results})
            if fallback:
                self.call_from_thread(
                    self._set_kernel_info,
                    f"{', '.join(kernels_used)} - shared-memory kernel unavailable: {fallback}",
                )
            else:
                self.call_from_thread(self._set_kernel_info, ", ".join(kernels_used))

            gpu_avg_time = sum(r["timings"]["total_s"] for r in gpu_results) / len(gpu_results)
            gpu_avg_throughput = sum(
                (r["n"] * r["m"]) / r["timings"]["total_s"] for r in gpu_results
            ) / len(gpu_results)
            total_cells = sum(r["n"] * r["m"] for r in gpu_results)
            self.call_from_thread(
                self._set_gpu_results, gpu_avg_time, gpu_avg_throughput, total_cells
            )

            per_gpu = {}
            for r in gpu_results:
                per_gpu[r["gpu_id"]] = per_gpu.get(r["gpu_id"], 0) + 1
            distribution = ", ".join(
                f"GPU {gid}: {count} pair(s)" for gid, count in sorted(per_gpu.items())
            )
            spread = max(per_gpu.values()) - min(per_gpu.values()) if per_gpu else 0
            balance = "even" if spread <= 1 else f"UNEVEN by {spread}"
            self.call_from_thread(
                self._set_dask_distribution,
                f"{distribution}  ({n_gpus_used} GPU(s), {balance})",
            )

            self.call_from_thread(self._set_status, "Benchmark complete")
            self.call_from_thread(self._set_current_file, "All chunks processed")
        finally:
            self._pipeline_running = False
            self._shutdown_dask()
            try:
                self.call_from_thread(self._set_dask_status, "shut down")
            except Exception:
                pass  # app already exiting


if __name__ == "__main__":
    GeneWeaverApp().run()