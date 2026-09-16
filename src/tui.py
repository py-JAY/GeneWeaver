import numpy as np

if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack

import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.containers import (
    Horizontal,
    Vertical,
    VerticalScroll,
)
from textual.widgets import (
    Header,
    Footer,
    Static,
    ProgressBar,
)
from textual.worker import get_current_worker

import GPUalgorithm as cuda_alignment
import offtarget


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_DIR = "data/chunks"

SAMPLE_SIZE = 500000

KERNEL_MODE = "warp"

GPU_POLL_SECONDS = 2.0

GUIDE_RNA = "GACCCCCTCCACCCCGCCTC"

MAX_MISMATCHES = 4

SCAN_BOTH_STRANDS = True

TOP_OFF_TARGETS = 5


# ============================================================
# GENEWEAVER APP
# ============================================================

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

    # ========================================================
    # UI
    # ========================================================

    def compose(self) -> ComposeResult:

        yield Header()

        with VerticalScroll(id="dashboard"):

            # ------------------------------------------------
            # PAGE TITLE
            # ------------------------------------------------

            with Vertical(id="hero"):

                yield Static(
                    "🧬  GENEWEAVER",
                    id="main-title",
                )

                yield Static(
                    "GPU-ACCELERATED CRISPR ALIGNMENT ENGINE",
                    id="main-subtitle",
                )

                yield Static(
                    "Genome processing • Global alignment • "
                    "GPU acceleration • Off-target analysis",
                    id="main-description",
                )

            # ------------------------------------------------
            # MAIN TWO COLUMN AREA
            # ------------------------------------------------

            with Horizontal(id="main-layout"):

                # =================================================
                # LEFT SIDEBAR
                # =================================================

                with Vertical(id="sidebar"):

                    # -----------------------------
                    # SCAN SETUP
                    # -----------------------------

                    with Vertical(classes="card setup-card"):

                        yield Static(
                            "SCAN SETUP",
                            classes="card-title",
                        )

                        yield Static(
                            "GENOME DATASET",
                            classes="field-label",
                        )

                        yield Static(
                            CHUNK_DIR,
                            classes="field-value",
                        )

                        yield Static(
                            "SAMPLE SIZE",
                            classes="field-label",
                        )

                        yield Static(
                            f"{SAMPLE_SIZE:,} bp",
                            id="sample-size",
                            classes="field-value",
                        )

                        yield Static(
                            "ALIGNMENT",
                            classes="field-label",
                        )

                        yield Static(
                            "Needleman-Wunsch",
                            classes="field-value",
                        )

                        yield Static(
                            "KERNEL",
                            classes="field-label",
                        )

                        yield Static(
                            "--",
                            id="sidebar-kernel",
                            classes="field-value",
                        )

                    # -----------------------------
                    # GENOME STATUS
                    # -----------------------------

                    with Vertical(classes="card"):

                        yield Static(
                            "GENOME STATUS",
                            classes="card-title",
                        )

                        with Horizontal(classes="metric-row"):
                            yield Static(
                                "TOTAL CHUNKS",
                                classes="metric-label",
                            )
                            yield Static(
                                "--",
                                id="metric-total-chunks",
                                classes="metric-value",
                            )

                        with Horizontal(classes="metric-row"):
                            yield Static(
                                "PROCESSED",
                                classes="metric-label",
                            )
                            yield Static(
                                "--",
                                id="metric-processed",
                                classes="metric-value",
                            )

                        with Horizontal(classes="metric-row"):
                            yield Static(
                                "CURRENT",
                                classes="metric-label",
                            )
                            yield Static(
                                "--",
                                id="metric-current",
                                classes="metric-value",
                            )

                    # -----------------------------
                    # PERFORMANCE
                    # -----------------------------

                    with Vertical(classes="card"):

                        yield Static(
                            "PERFORMANCE",
                            classes="card-title",
                        )

                        with Horizontal(classes="metric-row"):
                            yield Static(
                                "CPU",
                                classes="metric-label",
                            )
                            yield Static(
                                "--",
                                id="cpu-performance",
                                classes="accent-value",
                            )

                        with Horizontal(classes="metric-row"):
                            yield Static(
                                "GPU",
                                classes="metric-label",
                            )
                            yield Static(
                                "--",
                                id="gpu-performance",
                                classes="accent-value",
                            )

                        with Horizontal(classes="metric-row"):
                            yield Static(
                                "THROUGHPUT",
                                classes="metric-label",
                            )
                            yield Static(
                                "--",
                                id="sidebar-throughput",
                                classes="accent-value",
                            )

                    # -----------------------------
                    # PIPELINE STATUS
                    # -----------------------------

                    with Vertical(classes="card status-card"):

                        yield Static(
                            "PIPELINE STATUS",
                            classes="card-title",
                        )

                        yield Static(
                            "INITIALIZING",
                            id="pipeline-status",
                            classes="status-badge",
                        )

                        yield Static(
                            "Waiting for genome processing...",
                            id="status-detail",
                            classes="muted",
                        )

                # =================================================
                # MAIN CONTENT
                # =================================================

                with Vertical(id="content"):

                    # -----------------------------
                    # SCAN PROGRESS
                    # -----------------------------

                    with Vertical(classes="card progress-card"):

                        with Horizontal(classes="section-header"):
                            yield Static(
                                "SCAN PROGRESS",
                                classes="card-title",
                            )

                            yield Static(
                                "0%",
                                id="overall-percent",
                                classes="percent",
                            )

                        yield Static(
                            "Genome chunk processing",
                            id="progress-description",
                            classes="muted",
                        )

                        yield ProgressBar(
                            total=1,
                            show_eta=False,
                            id="chunk_progress",
                        )

                        with Horizontal(classes="progress-info"):
                            yield Static(
                                "Chunks: 0 / 0",
                                id="chunk_count",
                            )

                            yield Static(
                                "Current file: --",
                                id="current_file",
                            )

                    # -----------------------------
                    # DEVICE / DASK
                    # -----------------------------

                    with Horizontal(classes="two-column"):

                        with Vertical(classes="card half-card"):

                            yield Static(
                                "GPU & CUDA",
                                classes="card-title",
                            )

                            yield Static(
                                "GPU: --",
                                id="gpu-status",
                                classes="info-line",
                            )

                            yield Static(
                                "CUDA: --",
                                id="cuda-status",
                                classes="info-line",
                            )

                            yield Static(
                                "GPU Count: --",
                                id="gpu-count",
                                classes="info-line",
                            )

                            yield Static(
                                "Memory: --",
                                id="gpu-memory",
                                classes="info-line",
                            )

                            yield Static(
                                "Utilization: --",
                                id="gpu-utilization",
                                classes="info-line",
                            )

                            yield Static(
                                "Devices: --",
                                id="gpu-devices",
                                classes="info-line",
                            )

                        with Vertical(classes="card half-card"):

                            yield Static(
                                "DASK DISTRIBUTION",
                                classes="card-title",
                            )

                            yield Static(
                                "Cluster: --",
                                id="dask-status",
                                classes="info-line",
                            )

                            yield Static(
                                "Worker → GPU: --",
                                id="dask-pinning",
                                classes="info-line",
                            )

                            yield Static(
                                "Pair distribution: --",
                                id="dask-distribution",
                                classes="info-line",
                            )

                            yield Static(
                                "Kernel: --",
                                id="kernel-info",
                                classes="info-line",
                            )

                            yield Static(
                                "Occupancy: --",
                                id="occupancy-info",
                                classes="info-line",
                            )

                    # -----------------------------
                    # ALIGNMENT RESULTS
                    # -----------------------------

                    with Vertical(classes="card"):

                        yield Static(
                            "ALIGNMENT RESULTS",
                            classes="card-title",
                        )

                        with Horizontal(classes="result-grid"):

                            with Vertical(classes="result-box"):
                                yield Static(
                                    "CHUNK LENGTH",
                                    classes="result-label",
                                )
                                yield Static(
                                    "--",
                                    id="chunk-length",
                                    classes="result-number",
                                )

                            with Vertical(classes="result-box"):
                                yield Static(
                                    "VRAM / PAIR",
                                    classes="result-label",
                                )
                                yield Static(
                                    "--",
                                    id="vram-estimate",
                                    classes="result-number",
                                )

                            with Vertical(classes="result-box"):
                                yield Static(
                                    "AVG TIME",
                                    classes="result-label",
                                )
                                yield Static(
                                    "--",
                                    id="gpu-average-time",
                                    classes="result-number",
                                )

                            with Vertical(classes="result-box"):
                                yield Static(
                                    "THROUGHPUT",
                                    classes="result-label",
                                )
                                yield Static(
                                    "--",
                                    id="gpu-throughput",
                                    classes="result-number",
                                )

                            with Vertical(classes="result-box"):
                                yield Static(
                                    "TOTAL CELLS",
                                    classes="result-label",
                                )
                                yield Static(
                                    "--",
                                    id="gpu-total-cells",
                                    classes="result-number",
                                )

                    # -----------------------------
                    # ALIGNMENT PROGRESS
                    # -----------------------------

                    with Vertical(classes="card"):

                        with Horizontal(classes="section-header"):
                            yield Static(
                                "PAIR ALIGNMENT",
                                classes="card-title",
                            )

                            yield Static(
                                "0 / 0",
                                id="chunk-pair",
                                classes="percent",
                            )

                        yield Static(
                            "Distributed sequence alignment",
                            id="alignment-progress-label",
                            classes="muted",
                        )

                        yield ProgressBar(
                            total=1,
                            show_eta=False,
                            id="alignment_progress",
                        )

                    # -----------------------------
                    # OFF-TARGET
                    # -----------------------------

                    with Vertical(classes="card"):

                        yield Static(
                            "OFF-TARGET ANALYSIS",
                            classes="card-title",
                        )

                        yield Static(
                            "Guide: --",
                            id="guide-info",
                            classes="info-line",
                        )

                        yield Static(
                            "Scan: waiting...",
                            id="offtarget-status",
                            classes="info-line",
                        )

                        yield Static(
                            "",
                            id="offtarget-ranking",
                            classes="ranking",
                        )

                    # -----------------------------
                    # DNA COMPARISON
                    # -----------------------------

                    with Vertical(classes="card dna-card"):

                        yield Static(
                            "DNA SEQUENCE COMPARISON",
                            classes="card-title",
                        )

                        yield Static(
                            "Target / candidate alignment",
                            classes="muted",
                        )

                        yield Static(
                            "TARGET",
                            classes="sequence-label",
                        )

                        yield Static(
                            "5' — waiting for sequence — 3'",
                            id="target-sequence",
                            classes="dna-sequence",
                        )

                        yield Static(
                            "",
                            id="mismatch-line",
                            classes="mismatch-line",
                        )

                        yield Static(
                            "CANDIDATE",
                            classes="sequence-label",
                        )

                        yield Static(
                            "5' — waiting for sequence — 3'",
                            id="candidate-sequence",
                            classes="dna-sequence",
                        )

        yield Footer()

    # ========================================================
    # MOUNT
    # ========================================================

    def on_mount(self) -> None:

        self._refresh_gpu_status()

        self._set_kernel_info(
            self._kernel_label()
        )

        self.query_one("#sidebar-kernel", Static).update(
            self._kernel_label()
        )

        self._set_guide_info()

        # Automatically detect all chunks
        self.chunk_files = sorted(
            Path(CHUNK_DIR).glob("chunk_*.npy")
        )

        total = len(self.chunk_files)

        self.query_one(
            "#metric-total-chunks",
            Static,
        ).update(str(total))

        self.query_one(
            "#metric-processed",
            Static,
        ).update("0")

        self.query_one(
            "#metric-current",
            Static,
        ).update("--")

        self.query_one(
            "#chunk_progress",
            ProgressBar,
        ).update(
            total=max(total, 1),
            progress=0,
        )

        self.query_one(
            "#chunk_count",
            Static,
        ).update(
            f"Chunks: 0 / {total}"
        )

        if not self.chunk_files:

            self._set_status(
                "No genome chunks found."
            )

            self._set_pipeline_status(
                "ERROR",
                "No chunk_*.npy files detected.",
            )

            return

        self._set_status(
            f"Found {total} genome chunks. Starting pipeline..."
        )

        self._set_pipeline_status(
            "RUNNING",
            f"Processing {total} genome chunks",
        )

        self._pipeline_running = True

        self.run_pipeline_worker()

        self.poll_gpu_utilization()

    # ========================================================
    # UNMOUNT
    # ========================================================

    def on_unmount(self) -> None:
        self._shutdown_dask()

    # ========================================================
    # BASIC UI HELPERS
    # ========================================================

    def _set_status(self, text: str) -> None:

        self.query_one(
            "#status-detail",
            Static,
        ).update(text)

    def _set_pipeline_status(
        self,
        status: str,
        detail: str,
    ) -> None:

        self.query_one(
            "#pipeline-status",
            Static,
        ).update(status)

        self.query_one(
            "#status-detail",
            Static,
        ).update(detail)

    def _set_current_file(self, text: str) -> None:

        self.query_one(
            "#current_file",
            Static,
        ).update(
            f"Current file: {text}"
        )

    def _set_kernel_info(self, text: str) -> None:

        self.query_one(
            "#kernel-info",
            Static,
        ).update(
            f"Kernel: {text}"
        )

    # ========================================================
    # KERNEL LABEL
    # ========================================================

    def _kernel_label(self) -> str:

        try:

            sq = (
                f"{cuda_alignment.TILE_DIM}"
                f"x{cuda_alignment.TILE_DIM}"
                f" tile"
            )

            wt = (
                f"{cuda_alignment.TILE_ROWS}"
                f"x{cuda_alignment.TILE_COLS}"
                f" tile"
            )

        except Exception:

            sq = "32x32 tile"
            wt = "32x128 tile"

        return {

            "warp":
                f"warp-tile ({wt}, O(n+m) VRAM)",

            "banded":
                f"banded tiled ({sq}, O(n+m) VRAM)",

            "tiled":
                f"tiled shared-memory ({sq})",

            "diagonal":
                "diagonal full DP matrix",

        }.get(
            KERNEL_MODE,
            KERNEL_MODE,
        )

    # ========================================================
    # GUIDE INFORMATION
    # ========================================================

    def _set_guide_info(self) -> None:

        strands = (
            "both strands"
            if SCAN_BOTH_STRANDS
            else "forward strand"
        )

        self.query_one(
            "#guide-info",
            Static,
        ).update(
            f"Guide: 5'-{GUIDE_RNA}-3'   "
            f"({len(GUIDE_RNA)} bp)   "
            f"PAM: NGG/NAG   "
            f"≤ {MAX_MISMATCHES} mismatches   "
            f"{strands}"
        )

    # ========================================================
    # PROGRESS
    # ========================================================

    def _set_chunk_progress(
        self,
        done: int,
        total: int,
    ) -> None:

        total = max(total, 1)

        self.query_one(
            "#chunk_progress",
            ProgressBar,
        ).update(
            total=total,
            progress=done,
        )

        self.query_one(
            "#chunk_count",
            Static,
        ).update(
            f"Chunks: {done} / {total}"
        )

        self.query_one(
            "#metric-processed",
            Static,
        ).update(str(done))

        percent = int(
            (done / total) * 100
        )

        self.query_one(
            "#overall-percent",
            Static,
        ).update(
            f"{percent}%"
        )

    def _set_alignment_progress(
        self,
        done: int,
        total: int,
    ) -> None:

        total = max(total, 1)

        self.query_one(
            "#alignment_progress",
            ProgressBar,
        ).update(
            total=total,
            progress=done,
        )

        self.query_one(
            "#chunk-pair",
            Static,
        ).update(
            f"{done} / {total}"
        )

    def _set_alignment_label(
        self,
        text: str,
    ) -> None:

        self.query_one(
            "#alignment-progress-label",
            Static,
        ).update(text)

    # ========================================================
    # DASK
    # ========================================================

    def _set_dask_status(
        self,
        text: str,
    ) -> None:

        self.query_one(
            "#dask-status",
            Static,
        ).update(
            f"Cluster: {text}"
        )

    def _set_dask_pinning(
        self,
        text: str,
    ) -> None:

        self.query_one(
            "#dask-pinning",
            Static,
        ).update(
            f"Worker → GPU: {text}"
        )

    def _set_dask_distribution(
        self,
        text: str,
    ) -> None:

        self.query_one(
            "#dask-distribution",
            Static,
        ).update(
            f"Pair distribution: {text}"
        )

    # ========================================================
    # GPU STATUS
    # ========================================================

    def _refresh_gpu_status(self) -> None:

        try:

            info = (
                cuda_alignment
                .gpu_status_info()
            )

        except Exception as exc:

            self.query_one(
                "#gpu-status",
                Static,
            ).update(
                f"GPU: Probe failed"
            )

            self.query_one(
                "#cuda-status",
                Static,
            ).update(
                f"CUDA: {str(exc)[:50]}"
            )

            return

        self._set_gpu_status_widgets(info)

    def _set_gpu_status_widgets(
        self,
        info: dict,
    ) -> None:

        n_gpus = info.get(
            "count",
            0,
        )

        devices = (
            info.get("devices")
            or []
        )

        # -------------------------
        # GPU
        # -------------------------

        if info.get("simulator"):

            self.query_one(
                "#gpu-status",
                Static,
            ).update(
                "GPU: Simulator"
            )

            self.query_one(
                "#cuda-status",
                Static,
            ).update(
                "CUDA: Simulator mode"
            )

        elif info.get("available"):

            name = (
                info.get("device_name")
                or "GPU"
            )

            self.query_one(
                "#gpu-status",
                Static,
            ).update(
                f"GPU: Connected ({name})"
            )

            self.query_one(
                "#cuda-status",
                Static,
            ).update(
                "CUDA: Available"
            )

        else:

            self.query_one(
                "#gpu-status",
                Static,
            ).update(
                "GPU: Not connected"
            )

            self.query_one(
                "#cuda-status",
                Static,
            ).update(
                "CUDA: Not available"
            )

        # -------------------------
        # GPU COUNT
        # -------------------------

        if n_gpus == 0:

            self.query_one(
                "#gpu-count",
                Static,
            ).update(
                "GPU Count: 0"
            )

        else:

            label = (
                "GPU"
                if n_gpus == 1
                else "GPUs"
            )

            self.query_one(
                "#gpu-count",
                Static,
            ).update(
                f"GPU Count: {n_gpus} {label}"
            )

        # -------------------------
        # DEVICES
        # -------------------------

        if devices:

            lines = []

            for d in devices:

                name = d.get(
                    "name",
                    "Unknown",
                )

                mem = d.get(
                    "memory_total_gb"
                )

                if mem is not None:

                    lines.append(
                        f"{d.get('index', '?')}: "
                        f"{name} "
                        f"({mem:.1f} GB)"
                    )

                else:

                    lines.append(
                        f"{d.get('index', '?')}: "
                        f"{name}"
                    )

            self.query_one(
                "#gpu-devices",
                Static,
            ).update(
                "Devices: "
                + " | ".join(lines)
            )

        else:

            self.query_one(
                "#gpu-devices",
                Static,
            ).update(
                "Devices: none"
            )

        # -------------------------
        # MEMORY
        # -------------------------

        free = info.get(
            "memory_free_gb"
        )

        total = info.get(
            "memory_total_gb"
        )

        if (
            free is not None
            and total is not None
        ):

            self.query_one(
                "#gpu-memory",
                Static,
            ).update(
                f"Memory: {free:.2f} / "
                f"{total:.2f} GB free"
            )

        else:

            self.query_one(
                "#gpu-memory",
                Static,
            ).update(
                "Memory: N/A"
            )

        # -------------------------
        # UTILIZATION
        # -------------------------

        util = info.get(
            "utilization_pct"
        )

        if util is not None:

            self.query_one(
                "#gpu-utilization",
                Static,
            ).update(
                f"Utilization: {util:.0f}%"
            )

        else:

            self.query_one(
                "#gpu-utilization",
                Static,
            ).update(
                "Utilization: N/A"
            )

    # ========================================================
    # GPU UTILIZATION POLLING
    # ========================================================

    @work(
        thread=True,
        exclusive=False,
    )
    def poll_gpu_utilization(self) -> None:

        poll_worker = (
            get_current_worker()
        )

        while (
            not poll_worker.is_cancelled
            and self._pipeline_running
        ):

            try:

                devices = (
                    cuda_alignment
                    .list_gpu_devices()
                )

                self.call_from_thread(
                    self._set_utilization_only,
                    devices,
                )

            except Exception:
                pass

            time.sleep(
                GPU_POLL_SECONDS
            )

    def _set_utilization_only(
        self,
        devices,
    ) -> None:

        vals = [
            d["utilization_pct"]
            for d in devices
            if d.get("utilization_pct")
            is not None
        ]

        if not vals:
            return

        avg = (
            sum(vals) / len(vals)
        )

        self.query_one(
            "#gpu-utilization",
            Static,
        ).update(
            f"Utilization: {avg:.0f}%"
        )

    # ========================================================
    # MEMORY INFORMATION
    # ========================================================

    def _set_memory_info(
        self,
        bp: int,
        per_pair_bytes: int,
        free_bytes,
    ) -> None:

        self.query_one(
            "#chunk-length",
            Static,
        ).update(
            f"{bp:,} bp"
        )

        if (
            per_pair_bytes
            >= 1_000_000_000
        ):

            need = (
                f"{per_pair_bytes / 1e9:.2f} GB"
            )

        else:

            need = (
                f"{per_pair_bytes / 1e6:.2f} MB"
            )

        self.query_one(
            "#vram-estimate",
            Static,
        ).update(
            need
        )

    # ========================================================
    # GPU RESULTS
    # ========================================================

    def _set_gpu_results(
        self,
        avg_time_s: float,
        avg_throughput: float,
        total_cells: int,
    ) -> None:

        self.query_one(
            "#gpu-average-time",
            Static,
        ).update(
            f"{avg_time_s * 1000:.2f} ms"
        )

        self.query_one(
            "#gpu-throughput",
            Static,
        ).update(
            f"{avg_throughput:,.0f}"
        )

        self.query_one(
            "#gpu-total-cells",
            Static,
        ).update(
            f"{total_cells:,}"
        )

        self.query_one(
            "#sidebar-throughput",
            Static,
        ).update(
            f"{avg_throughput:,.0f}"
        )

        self.query_one(
            "#gpu-performance",
            Static,
        ).update(
            f"{avg_time_s * 1000:.1f} ms"
        )

    # ========================================================
    # CPU RESULT
    # ========================================================

    def _set_cpu_result(
        self,
        elapsed_s: float,
    ) -> None:

        self.query_one(
            "#cpu-performance",
            Static,
        ).update(
            f"{elapsed_s * 1000:.1f} ms"
        )

    # ========================================================
    # OFF TARGET
    # ========================================================

    def _set_offtarget_status(
        self,
        text: str,
    ) -> None:

        self.query_one(
            "#offtarget-status",
            Static,
        ).update(
            f"Scan: {text}"
        )

    def _set_offtarget_ranking(
        self,
        hits: list,
    ) -> None:

        try:

            rendered = (
                offtarget.render_ranking(
                    hits,
                    TOP_OFF_TARGETS,
                )
            )

        except Exception:

            rendered = (
                "No ranking available."
            )

        self.query_one(
            "#offtarget-ranking",
            Static,
        ).update(rendered)

        # Update DNA preview
        if hits:

            first = hits[0]

            guide = first.get(
                "guide",
                GUIDE_RNA,
            )

            site = first.get(
                "site",
                "",
            )

            self.query_one(
                "#target-sequence",
                Static,
            ).update(
                f"5' — {guide} — 3'"
            )

            self.query_one(
                "#candidate-sequence",
                Static,
            ).update(
                f"5' — {site} — 3'"
            )

            mismatches = first.get(
                "mismatches",
                first.get(
                    "mismatch_count",
                    0,
                ),
            )

            self.query_one(
                "#mismatch-line",
                Static,
            ).update(
                f"Mismatch count: {mismatches}"
            )

    # ========================================================
    # DASK SHUTDOWN
    # ========================================================

    def _shutdown_dask(self) -> None:

        client = self._dask_client
        cluster = self._dask_cluster

        self._dask_client = None
        self._dask_cluster = None

        for obj in (
            client,
            cluster,
        ):

            if obj is not None:

                try:
                    obj.close()

                except Exception:
                    pass

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    @work(thread=True)
    def run_pipeline_worker(self) -> None:

        worker = (
            get_current_worker()
        )

        # ----------------------------------------------------
        # GPU INFORMATION
        # ----------------------------------------------------

        gpu_info = (
            cuda_alignment
            .gpu_status_info()
        )

        self.call_from_thread(
            self._set_gpu_status_widgets,
            gpu_info,
        )

        n_gpus = max(
            1,
            gpu_info.get(
                "count",
                0,
            ),
        )

        # ----------------------------------------------------
        # START DASK
        # ----------------------------------------------------

        self.call_from_thread(
            self._set_status,
            f"Starting Dask cluster "
            f"with {n_gpus} worker(s)...",
        )

        try:

            client, cluster = (
                cuda_alignment
                .create_dask_cluster(
                    n_gpus
                )
            )

        except Exception as exc:

            self.call_from_thread(
                self._set_dask_status,
                f"failed: {exc}",
            )

            self.call_from_thread(
                self._set_pipeline_status,
                "ERROR",
                "Dask cluster failed",
            )

            self._pipeline_running = False

            return

        self._dask_client = client
        self._dask_cluster = cluster

        n_workers = (
            len(
                getattr(
                    cluster,
                    "workers",
                    {},
                )
            )
            or n_gpus
        )

        self.call_from_thread(
            self._set_dask_status,
            f"Running • {n_workers} worker(s)",
        )

        # ----------------------------------------------------
        # GPU PINNING
        # ----------------------------------------------------

        try:

            pin_map = (
                cuda_alignment
                .worker_gpu_map(
                    client,
                    n_gpus,
                )
            )

        except Exception:

            pin_map = []

        if pin_map:

            text = ", ".join(
                f"{addr.rsplit('/', 1)[-1]} "
                f"→ GPU {gid}"
                for addr, gid in pin_map
            )

            self.call_from_thread(
                self._set_dask_pinning,
                text,
            )

        else:

            self.call_from_thread(
                self._set_dask_pinning,
                "unavailable",
            )

        # ----------------------------------------------------
        # PARSE CHUNKS
        # ----------------------------------------------------

        try:

            self.call_from_thread(
                self._set_status,
                "Parsing genome chunks...",
            )

            def parse_progress_cb(
                done,
                total,
                filename,
            ):

                if worker.is_cancelled:
                    return

                self.call_from_thread(
                    self._set_current_file,
                    filename,
                )

                self.call_from_thread(
                    self._set_chunk_progress,
                    done,
                    total,
                )

                self.call_from_thread(
                    self._set_status,
                    f"Parsed chunk "
                    f"{done}/{total}",
                )

                self.call_from_thread(
                    self.query_one(
                        "#metric-current",
                        Static,
                    ).update,
                    filename,
                )

            sequences = (
                cuda_alignment
                .parse_paths_dask(
                    [
                        str(p)
                        for p in self.chunk_files
                    ],
                    SAMPLE_SIZE,
                    client=client,
                    progress_callback=
                    parse_progress_cb,
                )
            )

        except Exception as exc:

            self.call_from_thread(
                self._set_status,
                f"Genome parsing failed: {exc}",
            )

            self.call_from_thread(
                self._set_pipeline_status,
                "ERROR",
                "Genome parsing failed",
            )

            self._pipeline_running = False

            return

        if worker.is_cancelled:
            return

        # ----------------------------------------------------
        # OFF-TARGET SCAN
        # ----------------------------------------------------

        self.call_from_thread(
            self._set_pipeline_status,
            "SCANNING",
            "Searching for PAM sites and off-targets",
        )

        self.call_from_thread(
            self._set_offtarget_status,
            "running...",
        )

        try:

            def scan_progress_cb(
                done,
                total,
                label,
                found,
            ):

                self.call_from_thread(
                    self._set_offtarget_status,
                    f"{done}/{total} chunks • "
                    f"{found} hit(s)",
                )

            off_targets = (
                offtarget
                .scan_chunks_dask(
                    sequences,
                    [
                        p.name
                        for p in self.chunk_files
                    ],
                    GUIDE_RNA,
                    max_mismatches=
                    MAX_MISMATCHES,
                    client=client,
                    both_strands=
                    SCAN_BOTH_STRANDS,
                    progress_callback=
                    scan_progress_cb,
                )
            )

            by_level = {}

            for hit in off_targets:

                level = hit.get(
                    "severity",
                    "UNKNOWN",
                )

                by_level[level] = (
                    by_level.get(
                        level,
                        0,
                    ) + 1
                )

            summary = ", ".join(
                f"{by_level[k]} {k}"
                for k in (
                    "HIGH",
                    "MEDIUM",
                    "LOW",
                )
                if k in by_level
            )

            if not summary:
                summary = "none"

            self.call_from_thread(
                self._set_offtarget_status,
                f"{len(off_targets):,} "
                f"site(s) • {summary}",
            )

            self.call_from_thread(
                self._set_offtarget_ranking,
                off_targets,
            )

        except Exception as exc:

            self.call_from_thread(
                self._set_offtarget_status,
                f"failed: {exc}",
            )

        # ----------------------------------------------------
        # ALIGNMENT
        # ----------------------------------------------------

        if worker.is_cancelled:
            return

        lengths = [
            cuda_alignment
            .sequence_length(s)
            for s in sequences
        ]

        longest = max(lengths)

        # ----------------------------------------------------
        # MEMORY CHECK
        # ----------------------------------------------------

        try:

            fits, needed, free = (
                cuda_alignment
                .check_device_capacity(
                    longest,
                    longest,
                    KERNEL_MODE,
                )
            )

            self.call_from_thread(
                self._set_memory_info,
                longest,
                needed,
                free,
            )

        except Exception:

            fits = True

        try:

            self.call_from_thread(
                self._set_occupancy,
                longest,
            )

        except Exception:
            pass

        # ----------------------------------------------------
        # IMPORTANT:
        # If no real GPU exists, don't pretend that GPU
        # alignment succeeded.
        # ----------------------------------------------------

        if (
            not gpu_info.get("available")
            and not gpu_info.get("simulator")
        ):

            self.call_from_thread(
                self._set_pipeline_status,
                "CPU MODE",
                "CUDA GPU unavailable on this machine",
            )

        # ----------------------------------------------------
        # GPU ALIGNMENT
        # ----------------------------------------------------

        self.call_from_thread(
            self._set_alignment_label,
            "Alignment Progress (Dask)",
        )

        self.call_from_thread(
            self._set_alignment_progress,
            0,
            max(len(sequences) - 1, 1),
        )

        def gpu_progress_cb(
            done,
            total,
            pair_label,
        ):

            self.call_from_thread(
                self._set_alignment_progress,
                done,
                total,
            )

        try:

            gpu_results, n_gpus_used = (
                cuda_alignment
                .align_sequence_pairs_gpu_dask(
                    sequences,
                    client=client,
                    n_gpus=n_gpus,
                    progress_callback=
                    gpu_progress_cb,
                    kernel=KERNEL_MODE,
                )
            )

        except Exception as exc:

            self.call_from_thread(
                self._set_status,
                f"GPU alignment failed: {exc}",
            )

            self.call_from_thread(
                self._set_pipeline_status,
                "GPU FAILED",
                "Alignment could not complete on CUDA",
            )

            self._pipeline_running = False

            self._shutdown_dask()

            return

        # ----------------------------------------------------
        # GPU RESULTS
        # ----------------------------------------------------

        if gpu_results:

            kernels_used = sorted(
                {
                    r.get(
                        "kernel",
                        "unknown",
                    )
                    for r in gpu_results
                }
            )

            self.call_from_thread(
                self._set_kernel_info,
                ", ".join(
                    kernels_used
                ),
            )

            gpu_avg_time = (
                sum(
                    r["timings"]["total_s"]
                    for r in gpu_results
                )
                / len(gpu_results)
            )

            gpu_avg_throughput = (
                sum(
                    (
                        r["n"]
                        * r["m"]
                    )
                    / r["timings"]["total_s"]
                    for r in gpu_results
                )
                / len(gpu_results)
            )

            total_cells = sum(
                r["n"] * r["m"]
                for r in gpu_results
            )

            self.call_from_thread(
                self._set_gpu_results,
                gpu_avg_time,
                gpu_avg_throughput,
                total_cells,
            )

            # ------------------------------------------------
            # DASK BALANCE
            # ------------------------------------------------

            per_gpu = {}

            for r in gpu_results:

                gid = r.get(
                    "gpu_id",
                    0,
                )

                per_gpu[gid] = (
                    per_gpu.get(
                        gid,
                        0,
                    )
                    + 1
                )

            distribution = ", ".join(
                f"GPU {gid}: {count} pair(s)"
                for gid, count
                in sorted(
                    per_gpu.items()
                )
            )

            if per_gpu:

                spread = (
                    max(
                        per_gpu.values()
                    )
                    - min(
                        per_gpu.values()
                    )
                )

            else:

                spread = 0

            balance = (
                "balanced"
                if spread <= 1
                else f"spread {spread}"
            )

            self.call_from_thread(
                self._set_dask_distribution,
                f"{distribution} • "
                f"{n_gpus_used} GPU(s) • "
                f"{balance}",
            )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        self.call_from_thread(
            self._set_chunk_progress,
            len(self.chunk_files),
            len(self.chunk_files),
        )

        self.call_from_thread(
            self._set_current_file,
            "All chunks processed",
        )

        self.call_from_thread(
            self._set_pipeline_status,
            "COMPLETE",
            "Genome processing and analysis finished",
        )

        self.call_from_thread(
            self._set_status,
            "Pipeline completed successfully",
        )

        self._pipeline_running = False

        self._shutdown_dask()

        try:

            self.call_from_thread(
                self._set_dask_status,
                "shut down",
            )

        except Exception:
            pass

    # ========================================================
    # OCCUPANCY
    # ========================================================

    def _set_occupancy(
        self,
        bp: int,
    ) -> None:

        try:

            occ = (
                cuda_alignment
                .occupancy_report(
                    bp,
                    bp,
                    KERNEL_MODE,
                )
            )

        except Exception:

            return

        self.query_one(
            "#occupancy-info",
            Static,
        ).update(
            f"Occupancy: "
            f"{occ['threads_per_block']} thr/block • "
            f"{occ['warps_per_block']} warps • "
            f"{occ['shared_per_block_b']/1024:.1f} KB shared"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    GeneWeaverApp().run()