"""Tests for frozen candidate discovery, overlap handling and link policy."""
import os

import pytest

from image_pipeline.algorithms import core as alg
from image_pipeline.utils.image_io import write_image
from image_pipeline.config.loader import PipelineConfig
from image_pipeline.batch.executor import BatchExecutor, print_text_report
from image_pipeline.batch.discovery import (
    discover_candidates,
    make_output_predictor,
    make_output_signature,
    resolve_symlink_chain,
    resolve_output_realpath,
    verify_candidate,
    REASON_BATCH_REPORT,
    REASON_BROKEN_SYMLINK,
    REASON_DIRECTORY,
    REASON_DUPLICATE_ENTITY,
    REASON_PLANNED_OUTPUT,
    REASON_PIPELINE_PRODUCT,
    REASON_SYMLINK_LOOP,
    REASON_SYMLINK_OUTSIDE,
    REASON_SYMLINK_NOT_FILE,
    REASON_UNSUPPORTED_EXTENSION,
)


def _linear_pipeline(suffix='_processed', fmt='PNG'):
    cfg = PipelineConfig({
        'version': '1.0', 'name': 'p',
        'nodes': [
            {'id': 'in', 'type': 'input'},
            {'id': 'gray', 'type': 'grayscale'},
            {'id': 'out', 'type': 'output', 'params': {'suffix': suffix, 'format': fmt}},
        ],
        'edges': [{'from': 'in', 'to': 'gray'}, {'from': 'gray', 'to': 'out'}],
    })
    executor, _ = cfg.build_executor()
    return executor


def _output_nodes(executor):
    return executor.graph.get_output_nodes()


def _png(path, size=8):
    write_image(alg.generate_gradient_image(size, size), path, fmt='PNG')


def _reasons(discovery):
    return {item.relative_path: item.reason for item in discovery.skipped}


class TestSeparateDirectories:
    def test_normal_run_discovers_all_images(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        for n in ('a.png', 'b.jpg', 'c.bmp'):
            _png(os.path.join(in_dir, n))
        ex = _linear_pipeline()
        d = discover_candidates(in_dir, out_dir,
                                predict_outputs=make_output_predictor(_output_nodes(ex)),
                                recognize_product=make_output_signature(_output_nodes(ex)))
        assert not d.overlap
        assert sorted(c.relative_path for c in d.candidates) == ['a.png', 'b.jpg', 'c.bmp']
        assert d.skipped == []

    def test_repeated_separate_runs_keep_stable_set(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'a.png'))
        ex = _linear_pipeline()
        b1 = BatchExecutor(ex, in_dir, out_dir)
        r1 = b1.run()
        b1.write_report(r1)
        assert r1.succeeded == 1
        # A second run over the same args must not grow the candidate set.
        b2 = BatchExecutor(ex, in_dir, out_dir)
        d2 = b2.discover()
        assert d2.candidate_count == 1
        r2 = b2.run(d2)
        assert r2.succeeded == 1 and r2.skipped == 0

    def test_empty_input_dir(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'empty')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        ex = _linear_pipeline()
        d = discover_candidates(in_dir, out_dir,
                                predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert d.candidate_count == 0

    def test_missing_input_dir_raises(self, tmpdir_path):
        from image_pipeline.utils.types import ValidationError
        with pytest.raises(ValidationError):
            discover_candidates(os.path.join(tmpdir_path, 'nope'),
                                os.path.join(tmpdir_path, 'out'))


class TestOverlap:
    def test_same_dir_excludes_products_of_current_sources(self, tmpdir_path):
        d_dir = os.path.join(tmpdir_path, 'same')
        os.makedirs(d_dir)
        _png(os.path.join(d_dir, 'a.png'))
        _png(os.path.join(d_dir, 'a_processed.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            d_dir, d_dir,
            predict_outputs=make_output_predictor(_output_nodes(ex)),
            recognize_product=make_output_signature(_output_nodes(ex)))
        assert res.overlap
        assert [c.relative_path for c in res.candidates] == ['a.png']
        assert _reasons(res)['a_processed.png'] == REASON_PLANNED_OUTPUT

    def test_same_dir_repeated_runs_do_not_stack_suffixes(self, tmpdir_path):
        d_dir = os.path.join(tmpdir_path, 'same')
        os.makedirs(d_dir)
        _png(os.path.join(d_dir, 'a.png'))
        ex = _linear_pipeline()
        for _ in range(3):
            b = BatchExecutor(ex, d_dir, d_dir)
            disc = b.discover()
            assert [c.relative_path for c in disc.candidates] == ['a.png']
            report = b.run(disc)
            assert report.succeeded == 1
        files = sorted(os.listdir(d_dir))
        assert files == ['a.png', 'a_processed.png']

    def test_orphan_product_without_source_is_excluded(self, tmpdir_path):
        d_dir = os.path.join(tmpdir_path, 'orphan')
        os.makedirs(d_dir)
        # Only a leftover generated file; its source x.png is gone.
        _png(os.path.join(d_dir, 'x_processed.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            d_dir, d_dir,
            predict_outputs=make_output_predictor(_output_nodes(ex)),
            recognize_product=make_output_signature(_output_nodes(ex)))
        assert res.candidate_count == 0
        assert _reasons(res)['x_processed.png'] == REASON_PIPELINE_PRODUCT

    def test_double_stacked_artifact_is_excluded(self, tmpdir_path):
        d_dir = os.path.join(tmpdir_path, 'dbl')
        os.makedirs(d_dir)
        _png(os.path.join(d_dir, 'x_processed_processed.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            d_dir, d_dir,
            predict_outputs=make_output_predictor(_output_nodes(ex)),
            recognize_product=make_output_signature(_output_nodes(ex)))
        assert res.candidate_count == 0
        assert _reasons(res)['x_processed_processed.png'] == REASON_PIPELINE_PRODUCT

    def test_product_signature_ignored_with_separate_dirs(self, tmpdir_path):
        # A file carrying the output suffix is a perfectly normal input when
        # the output directory is elsewhere.
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'data_processed.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, out_dir,
            predict_outputs=make_output_predictor(_output_nodes(ex)),
            recognize_product=make_output_signature(_output_nodes(ex)))
        assert not res.overlap
        assert [c.relative_path for c in res.candidates] == ['data_processed.png']

    def test_batch_report_always_excluded(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'a.png'))
        with open(os.path.join(in_dir, 'batch_report.json'), 'w') as f:
            f.write('{}')
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, out_dir, predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert _reasons(res)['batch_report.json'] == REASON_BATCH_REPORT

    def test_output_subdir_overlap_is_flagged(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(in_dir, 'results')
        os.makedirs(out_dir)
        _png(os.path.join(in_dir, 'z.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, out_dir, predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert res.overlap
        assert [c.relative_path for c in res.candidates] == ['z.png']

    def test_empty_suffix_in_place_overwrite_is_refused(self, tmpdir_path):
        # Output node with empty suffix + out==in would overwrite the source.
        # The real-path planned-output comparison must exclude the source.
        d_dir = os.path.join(tmpdir_path, 'same')
        os.makedirs(d_dir)
        _png(os.path.join(d_dir, 'a.png'))
        ex = _linear_pipeline(suffix='')
        res = discover_candidates(
            d_dir, d_dir, predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert res.overlap
        assert res.candidate_count == 0
        assert _reasons(res)['a.png'] == REASON_PLANNED_OUTPUT


class TestMultipleOutputs:
    def test_all_output_names_excluded(self, tmpdir_path):
        d_dir = os.path.join(tmpdir_path, 'multi')
        os.makedirs(d_dir)
        cfg = PipelineConfig({
            'version': '1.0', 'name': 'm',
            'nodes': [
                {'id': 'in', 'type': 'input'},
                {'id': 'gray', 'type': 'grayscale'},
                {'id': 'o1', 'type': 'output', 'params': {'suffix': '_a', 'format': 'PNG'}},
                {'id': 'o2', 'type': 'output', 'params': {'suffix': '_b', 'format': 'JPEG'}},
            ],
            'edges': [{'from': 'in', 'to': 'gray'},
                      {'from': 'gray', 'to': 'o1'}, {'from': 'gray', 'to': 'o2'}],
        })
        ex, _ = cfg.build_executor()
        _png(os.path.join(d_dir, 'k.png'))
        _png(os.path.join(d_dir, 'k_a.png'))
        _png(os.path.join(d_dir, 'k_b.jpg'))
        res = discover_candidates(
            d_dir, d_dir,
            predict_outputs=make_output_predictor(_output_nodes(ex)),
            recognize_product=make_output_signature(_output_nodes(ex)))
        assert [c.relative_path for c in res.candidates] == ['k.png']
        reasons = _reasons(res)
        assert reasons['k_a.png'] == REASON_PLANNED_OUTPUT
        assert reasons['k_b.jpg'] == REASON_PLANNED_OUTPUT


class TestLinks:
    def test_symlink_outside_input_is_skipped(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        outside = os.path.join(tmpdir_path, 'outside')
        os.makedirs(in_dir)
        os.makedirs(outside)
        _png(os.path.join(outside, 'ext.png'))
        os.symlink(os.path.join(outside, 'ext.png'), os.path.join(in_dir, 'link.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, os.path.join(tmpdir_path, 'out'),
            predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert _reasons(res)['link.png'] == REASON_SYMLINK_OUTSIDE
        assert res.candidate_count == 0

    def test_symlink_loop_is_skipped(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        os.symlink('b.png', os.path.join(in_dir, 'a.png'))
        os.symlink('a.png', os.path.join(in_dir, 'b.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, os.path.join(tmpdir_path, 'out'),
            predict_outputs=make_output_predictor(_output_nodes(ex)))
        reasons = _reasons(res)
        assert reasons['a.png'] == REASON_SYMLINK_LOOP
        assert reasons['b.png'] == REASON_SYMLINK_LOOP

    def test_broken_symlink_is_skipped(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        os.symlink('missing.png', os.path.join(in_dir, 'broken.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, os.path.join(tmpdir_path, 'out'),
            predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert _reasons(res)['broken.png'] == REASON_BROKEN_SYMLINK

    def test_symlink_to_directory_is_skipped(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        outside = os.path.join(tmpdir_path, 'outside')
        os.makedirs(in_dir)
        os.makedirs(outside)
        os.symlink(outside, os.path.join(in_dir, 'dirlink'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, os.path.join(tmpdir_path, 'out'),
            predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert _reasons(res)['dirlink'] == REASON_SYMLINK_NOT_FILE

    def test_hard_links_to_one_entity_processed_once(self, tmpdir_path):
        # Both names are regular files; neither is more "real" than the other.
        # The deterministic tie-break is shortest relative path then lexical.
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'real.png'))
        os.link(os.path.join(in_dir, 'real.png'), os.path.join(in_dir, 'real_second.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, os.path.join(tmpdir_path, 'out'),
            predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert [c.relative_path for c in res.candidates] == ['real.png']
        assert _reasons(res)['real_second.png'] == REASON_DUPLICATE_ENTITY

    def test_symlink_alias_real_name_wins(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'real.png'))
        os.symlink('real.png', os.path.join(in_dir, 'alias.png'))
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, os.path.join(tmpdir_path, 'out'),
            predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert [c.relative_path for c in res.candidates] == ['real.png']
        assert _reasons(res)['alias.png'] == REASON_DUPLICATE_ENTITY

    def test_symlinked_output_dir_resolves_to_input(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'img.png'))
        _png(os.path.join(in_dir, 'img_processed.png'))
        out_link = os.path.join(tmpdir_path, 'out_link')
        os.symlink(in_dir, out_link)
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, out_link,
            predict_outputs=make_output_predictor(_output_nodes(ex)),
            recognize_product=make_output_signature(_output_nodes(ex)))
        assert res.overlap
        assert res.input_real == res.output_real
        assert [c.relative_path for c in res.candidates] == ['img.png']
        assert _reasons(res)['img_processed.png'] == REASON_PLANNED_OUTPUT


class TestMiscEntries:
    def test_subdirectory_and_unsupported_files_skipped(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(os.path.join(in_dir, 'sub'))
        os.makedirs(in_dir, exist_ok=True)
        _png(os.path.join(in_dir, 'a.png'))
        with open(os.path.join(in_dir, 'notes.txt'), 'w') as f:
            f.write('x')
        ex = _linear_pipeline()
        res = discover_candidates(
            in_dir, os.path.join(tmpdir_path, 'out'),
            predict_outputs=make_output_predictor(_output_nodes(ex)))
        reasons = _reasons(res)
        assert reasons['sub'] == REASON_DIRECTORY
        assert reasons['notes.txt'] == REASON_UNSUPPORTED_EXTENSION
        assert [c.relative_path for c in res.candidates] == ['a.png']


class TestFrozenVerification:
    def test_modified_file_after_freeze_is_rejected(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'a.png'))
        ex = _linear_pipeline()
        batch = BatchExecutor(ex, in_dir, out_dir)
        disc = batch.discover()
        cand = disc.candidates[0]
        # Replace content after the set was frozen.
        write_image(alg.generate_checkerboard(12, 12, 3),
                    os.path.join(in_dir, 'a.png'), fmt='PNG')
        msg = verify_candidate(cand, disc.planned_outputs)
        assert msg is not None
        assert 'changed' in msg or 'replaced' in msg
        report = batch.run(disc)
        result = report.results[0]
        assert not result.success
        assert 'frozen' in result.error

    def test_deleted_file_after_freeze_is_rejected(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'a.png'))
        ex = _linear_pipeline()
        batch = BatchExecutor(ex, in_dir, out_dir)
        disc = batch.discover()
        cand = disc.candidates[0]
        os.remove(os.path.join(in_dir, 'a.png'))
        msg = verify_candidate(cand, disc.planned_outputs)
        assert msg is not None and 'removed' in msg

    def test_unchanged_file_passes_verification(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'a.png'))
        ex = _linear_pipeline()
        disc = discover_candidates(
            in_dir, out_dir, predict_outputs=make_output_predictor(_output_nodes(ex)))
        assert verify_candidate(disc.candidates[0], disc.planned_outputs) is None


class TestReceipt:
    def test_report_lists_skipped_relative_paths_and_reasons(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        out_dir = os.path.join(tmpdir_path, 'out')
        os.makedirs(in_dir)
        os.symlink('missing.png', os.path.join(in_dir, 'broken.png'))
        _png(os.path.join(in_dir, 'a.png'))
        ex = _linear_pipeline()
        batch = BatchExecutor(ex, in_dir, out_dir)
        report = batch.run()
        assert report.skipped == 1
        item = report.skipped_items[0]
        assert item.relative_path == 'broken.png'
        assert item.reason == REASON_BROKEN_SYMLINK
        data = report.to_dict()
        assert data['summary']['skipped'] == 1
        assert data['skipped'][0]['path'] == 'broken.png'
        assert data['skipped'][0]['reason'] == REASON_BROKEN_SYMLINK
        text = print_text_report(report)
        assert 'broken.png' in text
        assert REASON_BROKEN_SYMLINK in text

    def test_overlap_warning_recorded(self, tmpdir_path):
        d_dir = os.path.join(tmpdir_path, 'same')
        os.makedirs(d_dir)
        _png(os.path.join(d_dir, 'a.png'))
        ex = _linear_pipeline()
        report = BatchExecutor(ex, d_dir, d_dir).run()
        assert any('overlap' in w for w in report.warnings)


class TestSymlinkChainResolver:
    def test_detects_self_loop_directly(self, tmpdir_path):
        loop = os.path.join(tmpdir_path, 'loop')
        os.symlink('loop', loop)
        target, reason = resolve_symlink_chain(loop)
        assert target is None
        assert reason == REASON_SYMLINK_LOOP

    def test_nonexistent_output_realpath(self, tmpdir_path):
        out = os.path.join(tmpdir_path, 'a', 'b', 'c')
        resolved = resolve_output_realpath(out)
        assert resolved == os.path.normpath(out)
