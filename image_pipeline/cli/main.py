import argparse
import json
import os
import sys
from typing import List, Optional
from .. import __version__
from ..config.loader import load_config_file, validate_config_file
from ..batch.executor import BatchExecutor, print_text_report
from ..utils.sample_generator import generate_all
from ..batch.discovery import discover_candidates, make_output_predictor, make_output_signature
from ..utils.types import ValidationError
from ..utils.image_io import predict_output_filename

def _print_skipped(skipped, stream, indent='  '):
    for item in skipped:
        print(f'{indent}[skipped:{item.reason}] {item.relative_path}', file=stream)
        print(f'{indent}    {item.message}', file=stream)

def cmd_run(args: argparse.Namespace) -> int:
    config_path = os.path.abspath(args.config)
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    try:
        cfg = load_config_file(config_path)
    except Exception as e:
        print(f'ERROR: Failed to load pipeline config: {e}', file=sys.stderr)
        return 3
    executor, validation = cfg.build_executor()
    if not validation.valid:
        print('ERROR: Pipeline validation failed:', file=sys.stderr)
        for issue in validation.errors:
            print(f'  - {issue}', file=sys.stderr)
        for issue in validation.warnings:
            print(f'  * {issue}', file=sys.stderr)
        return 3
    if validation.warnings and (not args.quiet):
        print('Pipeline warnings:')
        for issue in validation.warnings:
            print(f'  * {issue}')
        print()
    if not args.quiet:
        print(f'Pipeline: {cfg.name}')
        print(f'Input dir: {input_dir}')
        print(f'Output dir: {output_dir}')
        print(f"Execution order: {' -> '.join(executor.execution_order)}")
        print()
    batch = BatchExecutor(executor, input_dir, output_dir, config_file=config_path)
    # Freeze the candidate set exactly once, before the output directory is
    # created and before any pixel is written; run() consumes this same
    # snapshot, so two scans can never disagree.
    try:
        discovery = batch.discover()
    except ValidationError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 3
    if not args.quiet:
        if discovery.overlap:
            print('WARNING: input and output directories resolve to overlapping locations.')
            print(f'  input (real) : {discovery.input_real}')
            print(f'  output (real): {discovery.output_real}')
            print('  Files this pipeline generates will be excluded from the inputs.')
            print()
        if discovery.skipped:
            print(f'{len(discovery.skipped)} item(s) excluded from this run:')
            _print_skipped(discovery.skipped, sys.stdout)
            print()
    if discovery.candidate_count == 0:
        product_skips = [item for item in discovery.skipped
                         if item.reason in ('planned_output', 'pipeline_product')]
        if product_skips:
            print('ERROR: No images to process: every image in the input directory is a file this '
                  'pipeline would (re)generate (output directory overlaps input). Refusing to run '
                  'to avoid re-processing generated files. Choose a separate output directory.',
                  file=sys.stderr)
            if args.quiet:
                _print_skipped(discovery.skipped, sys.stderr)
        else:
            print('ERROR: No supported images found after applying exclusions '
                  f'(input: {input_dir}).', file=sys.stderr)
            if args.quiet and discovery.skipped:
                _print_skipped(discovery.skipped, sys.stderr)
        return 3
    if not args.quiet:
        print(f'Frozen {discovery.candidate_count} candidate image(s) for this run.')
    progress_cb = None
    if not args.quiet and (not args.no_progress):

        def _progress(done: int, total: int, result):
            pct = 100.0 * done / total
            status = 'OK' if result.success else 'FAIL'
            fname = os.path.basename(result.input_path)
            bar_len = 30
            filled = int(bar_len * done // total)
            bar = '#' * filled + '-' * (bar_len - filled)
            sys.stdout.write(f'\r  [{bar}] {done}/{total} ({pct:5.1f}%) Last: [{status}] {fname}     ')
            sys.stdout.flush()
            if done == total:
                sys.stdout.write('\n')
        progress_cb = _progress
    batch.progress_callback = progress_cb
    report = batch.run(discovery)
    if progress_cb is not None:
        sys.stdout.write('\n')
    if not args.no_report:
        report_path = batch.write_report(report)
        if not args.quiet:
            print(f'JSON report written to: {report_path}')
    verbose = args.verbose
    text_report = print_text_report(report, verbose=verbose)
    if not args.quiet:
        print()
        print(text_report)
    if report.succeeded == report.total:
        return 0
    elif report.succeeded > 0 and report.failed > 0:
        return 1
    else:
        return 2

def cmd_validate(args: argparse.Namespace) -> int:
    config_path = os.path.abspath(args.config)
    if not args.quiet:
        print(f'Validating: {config_path}')
        print()
    result = validate_config_file(config_path)
    if args.quiet:
        return 0 if result.valid else 1
    print(str(result))
    print()
    if result.valid:
        try:
            cfg = load_config_file(config_path)
            executor, _ = cfg.build_executor()
            print('Execution order:')
            for i, nid in enumerate(executor.execution_order, 1):
                node = executor.graph.nodes[nid]
                params_str = json.dumps(node.effective_params(), sort_keys=True) if node.effective_params() else '{}'
                print(f'  {i}. [{node.node_type.value}] {nid}  params={params_str}')
        except Exception:
            pass
        print()
        print('Result: VALID')
        return 0
    else:
        print('Result: INVALID')
        return 1

def cmd_generate_sample(args: argparse.Namespace) -> int:
    base_dir = os.path.abspath(args.output_dir)
    if not args.quiet:
        print(f'Generating sample data into: {base_dir}')
    info = generate_all(base_dir)
    if not args.quiet:
        print()
        print(f"Generated {info['image_count']} sample images in:")
        print(f"  {info['input_dir']}")
        for p in info['images']:
            print(f'    - {os.path.basename(p)}')
        print()
        print(f"Generated {info['config_count']} pipeline configs in:")
        print(f"  {info['pipeline_dir']}")
        for p in info['configs']:
            name = os.path.basename(p)
            tag = ''
            if '_INVALID' in name:
                tag = ' (intentionally invalid - for testing validate)'
            print(f'    - {name}{tag}')
        print()
        print(f"Output directory (ready for 'run' command):")
        print(f"  {info['output_dir']}")
        print()
        print('Try running:')
        simple_pipe = os.path.join(info['pipeline_dir'], 'pipeline_simple.json')
        print(f"  imgpipe run --config {simple_pipe} --input {info['input_dir']} --output {info['output_dir']}")
    return 0

def cmd_dry_run(args: argparse.Namespace) -> int:
    config_path = os.path.abspath(args.config)
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    try:
        cfg = load_config_file(config_path)
    except Exception as e:
        print(f'ERROR: Failed to load pipeline config: {e}', file=sys.stderr)
        return 2
    executor, validation = cfg.build_executor()
    if not validation.valid:
        print('ERROR: Pipeline validation failed:', file=sys.stderr)
        for issue in validation.errors:
            print(f'  - {issue}', file=sys.stderr)
        return 3
    print(f'=== DRY RUN ===')
    print(f'Pipeline name   : {cfg.name}')
    print(f'Config file     : {config_path}')
    print(f'Input directory : {input_dir}')
    print(f'Output directory: {output_dir}')
    print()
    print('--- Node Execution Order ---')
    for i, nid in enumerate(executor.execution_order, 1):
        node = executor.graph.nodes[nid]
        upstream = executor.graph.upstream.get(nid, [])
        if upstream:
            src_list = ', '.join((f'{src}->slot{slot}' for src, slot in sorted(upstream, key=lambda x: x[1])))
            deps = f'  (depends on: {src_list})'
        else:
            deps = '  (source node)'
        params_str = json.dumps(node.effective_params(), sort_keys=True) if node.effective_params() else '{}'
        print(f'  {i:2d}. [{node.node_type.value:15s}] {nid:20s} params={params_str}{deps}')
    print()
    if os.path.isdir(input_dir):
        output_nodes_for_discovery = executor.graph.get_output_nodes()
        discovery = discover_candidates(
            input_dir,
            output_dir,
            predict_outputs=make_output_predictor(output_nodes_for_discovery),
            recognize_product=make_output_signature(output_nodes_for_discovery),
        )
        images = [c.path for c in discovery.candidates]
    else:
        discovery = None
        images = []
    print(f'--- Input Images ({len(images)} selected) ---')
    if not images:
        if not os.path.isdir(input_dir):
            print(f'  (input directory does not exist: {input_dir})')
        else:
            print('  (no eligible images after exclusions)')
    else:
        for c in discovery.candidates:
            sz = ''
            try:
                from ..utils.image_io import image_size
                w, h = image_size(c.path)
                sz = f'  ({w}x{h})'
            except Exception:
                pass
            print(f'  - {c.relative_path}{sz}')
    if discovery is not None and discovery.skipped:
        print()
        print(f'--- Excluded From Input ({len(discovery.skipped)}) ---')
        _print_skipped(discovery.skipped, sys.stdout)
    if discovery is not None and discovery.overlap:
        print()
        print('NOTE: input and output directories overlap; generated files are excluded from inputs.')
    print()
    print('--- Predicted Output Files ---')
    output_nodes = executor.graph.get_output_nodes()
    if not output_nodes:
        print('  (no output nodes in pipeline - no files will be written)')
    elif not images:
        print('  (no eligible input images - no output files predicted)')
    else:
        for c in discovery.candidates:
            fname = c.relative_path
            base = os.path.basename(fname)
            for onode in output_nodes:
                params = onode.effective_params()
                suffix = params.get('suffix', '') or ''
                fmt = params.get('format')
                out_name = predict_output_filename(base, suffix=suffix, fmt=fmt)
                out_path = os.path.join(output_dir, out_name)
                print(f'  {fname} + [{onode.node_id}] -> {out_path}')
    print()
    print('--- Summary ---')
    print(f'  Nodes           : {len(executor.execution_order)}')
    print(f'  Edges           : {sum((len(v) for v in executor.graph.upstream.values()))}')
    print(f'  Images to process: {len(images)}')
    print(f'  Output nodes    : {len(output_nodes)}')
    print(f'  Total output files (max): {len(images) * len(output_nodes)}')
    print()
    print('Dry run complete. No files were processed or written.')
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='imgpipe', description='Image batch processing CLI with node-graph pipelines. All pixel operations are implemented manually (no Pillow filter shortcuts).')
    parser.add_argument('--version', action='version', version=f'imgpipe {__version__}')
    sub = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')
    p_run = sub.add_parser('run', help='Execute a pipeline over images in an input directory.', description='Execute pipeline and write outputs to target directory.')
    p_run.add_argument('-c', '--config', required=True, help='Path to pipeline JSON config file.')
    p_run.add_argument('-i', '--input', '--input-dir', required=True, dest='input_dir', help='Directory containing input images.')
    p_run.add_argument('-o', '--output', '--output-dir', required=True, dest='output_dir', help='Directory for processed output images.')
    p_run.add_argument('-q', '--quiet', action='store_true', help='Suppress non-error output.')
    p_run.add_argument('-v', '--verbose', action='store_true', help='Include per-node details in text report.')
    p_run.add_argument('--no-progress', action='store_true', help='Disable progress bar during processing.')
    p_run.add_argument('--no-report', action='store_true', help='Do not write JSON batch report file.')
    p_run.set_defaults(func=cmd_run)
    p_val = sub.add_parser('validate', help='Validate a pipeline config (params, connections, no cycles).', description='Check pipeline config for errors and warnings.')
    p_val.add_argument('-c', '--config', required=True, help='Path to pipeline JSON config file.')
    p_val.add_argument('-q', '--quiet', action='store_true', help='Suppress output; only set exit code (0=valid, 1=invalid).')
    p_val.set_defaults(func=cmd_validate)
    p_gen = sub.add_parser('generate-sample', help='Generate sample images and sample pipeline config files.', description='Create constructed test images and valid/invalid pipeline configurations for testing and learning.')
    p_gen.add_argument('-o', '--output-dir', required=False, default='./sample_data', help='Directory to place generated data (default: ./sample_data).')
    p_gen.add_argument('-q', '--quiet', action='store_true', help='Suppress summary output.')
    p_gen.set_defaults(func=cmd_generate_sample)
    p_dry = sub.add_parser('dry-run', help='List execution order, input images, and predicted output filenames without processing anything.', description='Show what would be done; do not read/write image pixels.')
    p_dry.add_argument('-c', '--config', required=True, help='Path to pipeline JSON config file.')
    p_dry.add_argument('-i', '--input', '--input-dir', required=True, dest='input_dir', help='Directory of input images.')
    p_dry.add_argument('-o', '--output', '--output-dir', required=True, dest='output_dir', help='Intended output directory (for predicting filenames).')
    p_dry.set_defaults(func=cmd_dry_run)
    return parser

def main(argv: Optional[List[str]]=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
if __name__ == '__main__':
    sys.exit(main())
