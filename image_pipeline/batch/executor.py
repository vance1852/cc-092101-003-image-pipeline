from typing import Any, Dict, List, Optional, Callable
import os
import json
import time
import traceback
from ..pipeline.engine import PipelineExecutor
from ..utils.types import BatchReport, ImageProcessingResult, ValidationError
from ..utils.image_io import is_valid_image
from ..utils.discovery import Candidate, DiscoveryResult, discover_inputs


class BatchExecutor:

    def __init__(self, pipeline_executor: PipelineExecutor, input_dir: str, output_dir: str, config_file: str='', progress_callback: Optional[Callable]=None):
        self.executor = pipeline_executor
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config_file = config_file
        self.progress_callback = progress_callback

    def _output_specs(self) -> List[Dict[str, Any]]:
        """Suffix/format of every output node in execution order."""
        specs = []
        try:
            nodes = self.executor.graph.get_output_nodes()
        except Exception:
            nodes = []
        for node in nodes:
            params = node.effective_params()
            specs.append({'suffix': params.get('suffix', ''), 'format': params.get('format')})
        return specs

    def _ensure_output_dir(self) -> None:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def discover(self) -> DiscoveryResult:
        """Freeze this run's candidate set before anything is written."""
        return discover_inputs(self.input_dir, self.output_dir, self._output_specs())

    def _recheck_candidate(self, candidate: Candidate) -> Optional[str]:
        """Verify a frozen candidate is still the same file before processing.

        Returns an error message if the file disappeared, changed identity or
        was modified after the candidate set was frozen, so a check/execute
        race can never result in silently processing the wrong version.
        """
        # Verify through the discovered name (which may itself be a symlink),
        # not only the resolved target: replacing the link with a new target
        # must be detected too.
        try:
            st = os.stat(candidate.display_path)
        except FileNotFoundError:
            return 'file removed between discovery and execution'
        except OSError as e:
            return f'file became unreadable between discovery and execution: {e}'
        real = os.path.realpath(candidate.display_path)
        if real != candidate.path:
            return 'symlink target changed between discovery and execution'
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != candidate.fingerprint():
            return 'file modified between discovery and execution'
        return None

    def run(self, discovery: Optional[DiscoveryResult]=None) -> BatchReport:
        report = BatchReport(pipeline_config_file=self.config_file, input_dir=self.input_dir, output_dir=self.output_dir)
        overall_start = time.perf_counter()

        # 1) Freeze the candidate set FIRST, while the output directory does
        #    not yet contain files from this run.  A snapshot prepared by the
        #    caller (e.g. the CLI preview) is reused so the preview and the
        #    execution can never disagree.
        try:
            if discovery is None:
                discovery = self.discover()
        except (ValidationError, FileNotFoundError, NotADirectoryError, OSError) as e:
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=str(e))
            report.results.append(dummy)
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report

        report.skipped_items.extend(discovery.skipped)
        report.skipped = len(report.skipped_items)

        # 2) Only now create the output directory.
        try:
            self._ensure_output_dir()
        except Exception as e:
            dummy = ImageProcessingResult(input_path='', output_path=None, success=False, error=f'Failed to create output directory: {e}')
            report.results.append(dummy)
            report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
            return report

        # 3) Process exactly the frozen candidates.
        candidates: List[Candidate] = discovery.candidates
        report.total = len(candidates)
        for idx, candidate in enumerate(candidates):
            img_path = candidate.path
            # Output naming uses the name the file was discovered under.
            filename = os.path.basename(candidate.display_path)
            img_result: ImageProcessingResult

            changed = self._recheck_candidate(candidate)
            if changed is not None:
                img_result = ImageProcessingResult(
                    input_path=candidate.rel_path, success=False,
                    error=f'Not processed: {changed}. Re-run the batch to pick up the current version.')
                report.failed += 1
                report.results.append(img_result)
                if self.progress_callback:
                    try:
                        self.progress_callback(idx + 1, report.total, img_result)
                    except Exception:
                        pass
                continue

            if not is_valid_image(img_path):
                img_result = ImageProcessingResult(input_path=candidate.rel_path, success=False, error='Image failed pre-check verification (likely corrupt or unsupported format)')
                report.failed += 1
                report.results.append(img_result)
                if self.progress_callback:
                    try:
                        self.progress_callback(idx + 1, report.total, img_result)
                    except Exception:
                        pass
                continue
            context: Dict[str, Any] = {'input_path': img_path, 'input_filename': filename, 'output_dir': self.output_dir, 'image_index': idx}
            try:
                img_result = self.executor.run(context)
                if img_result.success:
                    report.succeeded += 1
                else:
                    report.failed += 1
            except Exception as e:
                img_result = ImageProcessingResult(input_path=img_path, success=False, error=f'Unexpected error during execution: {e}\n{traceback.format_exc()}')
                report.failed += 1
            # Report inputs using the stable relative path of the candidate.
            img_result.input_path = candidate.rel_path
            report.results.append(img_result)
            if self.progress_callback:
                try:
                    self.progress_callback(idx + 1, report.total, img_result)
                except Exception:
                    pass
        report.total_duration_ms = (time.perf_counter() - overall_start) * 1000.0
        return report

    def write_report(self, report: BatchReport, path: str=None) -> str:
        if path is None:
            path = os.path.join(self.output_dir, 'batch_report.json')
        report_dir = os.path.dirname(path)
        if report_dir and (not os.path.exists(report_dir)):
            os.makedirs(report_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

def print_text_report(report: BatchReport, verbose: bool=False) -> str:
    lines = []
    lines.append('=' * 60)
    lines.append('BATCH PROCESSING REPORT')
    lines.append('=' * 60)
    lines.append(f'Pipeline config : {report.pipeline_config_file}')
    lines.append(f'Input directory : {report.input_dir}')
    lines.append(f'Output directory: {report.output_dir}')
    lines.append('')
    lines.append('--- Summary ---')
    lines.append(f'Total images    : {report.total}')
    lines.append(f'Succeeded       : {report.succeeded}')
    lines.append(f'Failed          : {report.failed}')
    lines.append(f'Skipped         : {report.skipped}')
    lines.append(f'Total duration  : {report.total_duration_ms:.2f} ms')
    if report.total > 0:
        lines.append(f'Avg per image   : {report.total_duration_ms / report.total:.2f} ms')
    lines.append('')
    if report.skipped_items:
        lines.append('--- Skipped Items ---')
        for s in report.skipped_items:
            detail = f' ({s.detail})' if s.detail else ''
            lines.append(f'  [SKIP] {s.rel_path}')
            lines.append(f'         Reason: {s.reason}{detail}')
        lines.append('')
    if verbose:
        lines.append('--- Per-Image Details ---')
        for r in report.results:
            status = 'OK' if r.success else 'FAIL'
            out = r.output_path or '(no output)'
            err = f'\n    ERROR: {r.error}' if r.error else ''
            lines.append(f'  [{status}] {r.input_path} -> {out} ({r.duration_ms:.2f} ms){err}')
            if verbose and r.node_results:
                for nr in r.node_results:
                    nstatus = 'OK' if nr.success else 'FAIL'
                    size = f'{nr.output_size[0]}x{nr.output_size[1]}' if nr.output_size else '?'
                    nerr = f' -> {nr.error}' if nr.error else ''
                    lines.append(f'      + {nstatus} {nr.node_id} ({nr.node_type}, {size}, {nr.duration_ms:.2f} ms){nerr}')
        lines.append('')
    if report.failed > 0:
        lines.append('--- Failed Images ---')
        for r in report.results:
            if not r.success:
                lines.append(f'  {r.input_path}')
                lines.append(f'    Reason: {r.error}')
        lines.append('')
    return '\n'.join(lines)
