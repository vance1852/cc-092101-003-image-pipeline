import json
import os
import shutil
import time

import pytest

from image_pipeline.utils.discovery import (
    discover_inputs, planned_output_names, is_under,
)
from image_pipeline.utils.image_io import write_image
from image_pipeline.algorithms import core as alg
from image_pipeline.config.loader import PipelineConfig, sample_pipeline_config
from image_pipeline.batch.executor import BatchExecutor


SPECS = [{'suffix': '_processed', 'format': None}]


def _png(path, size=8):
    write_image(alg.generate_gradient_image(size, size), path, fmt='PNG')


def _make_pipeline(suffix='_processed'):
    cfg = PipelineConfig(sample_pipeline_config())
    executor, _ = cfg.build_executor()
    return executor


@pytest.fixture
def work(tmpdir_path):
    in_dir = os.path.join(tmpdir_path, 'in')
    out_dir = os.path.join(tmpdir_path, 'out')
    os.makedirs(in_dir)
    return in_dir, out_dir


class TestNormalSeparatedDirs:

    def test_finds_all_images(self, work):
        in_dir, out_dir = work
        for n in ('a.png', 'b.jpg', 'c.bmp'):
            _png(os.path.join(in_dir, n))
        d = discover_inputs(in_dir, out_dir, SPECS)
        assert sorted(c.rel_path for c in d.candidates) == ['a.png', 'b.jpg', 'c.bmp']
        assert d.skipped == []

    def test_ignores_non_images(self, work):
        in_dir, out_dir = work
        _png(os.path.join(in_dir, 'a.png'))
        with open(os.path.join(in_dir, 'notes.txt'), 'w') as f:
            f.write('x')
        d = discover_inputs(in_dir, out_dir, SPECS)
        assert [c.rel_path for c in d.candidates] == ['a.png']
        assert d.skipped == []

    def test_run_unchanged_separated_dirs(self, work):
        in_dir, out_dir = work
        _png(os.path.join(in_dir, 'a.png'))
        batch = BatchExecutor(_make_pipeline(), in_dir, out_dir, 'cfg')
        report = batch.run()
        assert report.total == 1
        assert report.succeeded == 1
        assert report.skipped == 0
        assert os.path.isfile(os.path.join(out_dir, 'a_processed.png'))


class TestOutputOverlapsInput:

    def test_same_dir_second_run_does_not_grow_set(self, work):
        in_dir, _ = work
        out_dir = in_dir  # output set into input
        batch = BatchExecutor(_make_pipeline(), in_dir, out_dir, 'cfg')
        r1 = batch.run()
        assert r1.total == 0
        batch.write_report(r1)

        d1 = discover_inputs(in_dir, out_dir,
                             batch._output_specs())
        # run 2 with an actual input present
        _png(os.path.join(in_dir, 'photo.png'))
        r2 = BatchExecutor(_make_pipeline(), in_dir, out_dir, 'cfg').run()
        assert r2.total == 1
        assert r2.succeeded == 1
        assert os.path.isfile(os.path.join(in_dir, 'photo_processed.png'))

        # run 3: yesterday's output + report must not be rediscovered
        d3 = discover_inputs(in_dir, out_dir,
                             BatchExecutor(_make_pipeline(), in_dir, out_dir)._output_specs())
        rels = sorted(c.rel_path for c in d3.candidates)
        assert rels == ['photo.png']
        reasons = {s.rel_path: s.reason for s in d3.skipped}
        assert 'photo_processed.png' in reasons
        assert 'batch_report.json' in reasons

    def test_repeated_runs_stable_candidate_count(self, work):
        in_dir, _ = work
        _png(os.path.join(in_dir, 'x.png'))
        for _ in range(3):
            BatchExecutor(_make_pipeline(), in_dir, in_dir, 'cfg').run()
        d = discover_inputs(in_dir, in_dir, SPECS)
        assert sorted(c.rel_path for c in d.candidates) == ['x.png']
        # x_processed.png excluded; its own suffix-stacking variants never appear
        names = set(os.listdir(in_dir))
        assert 'x_processed_processed.png' not in names

    def test_output_subdir_of_input(self, work):
        in_dir, _ = work
        out_dir = os.path.join(in_dir, 'processed')
        _png(os.path.join(in_dir, 'x.png'))
        # Simulate a previous output and rerun: the scan is non-recursive so
        # files in the output subdir are never scanned.
        os.makedirs(out_dir)
        _png(os.path.join(out_dir, 'x_processed.png'))
        d = discover_inputs(in_dir, out_dir, SPECS)
        assert [c.rel_path for c in d.candidates] == ['x.png']

    def test_input_subdir_of_output(self, tmpdir_path):
        out_dir = os.path.join(tmpdir_path, 'out')
        in_dir = os.path.join(out_dir, 'incoming')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'x.png'))
        d = discover_inputs(in_dir, out_dir, SPECS)
        assert [c.rel_path for c in d.candidates] == ['x.png']

    def test_report_excluded_from_discovery(self, work):
        in_dir, _ = work
        out_dir = in_dir
        _png(os.path.join(in_dir, 'a.png'))
        with open(os.path.join(in_dir, 'batch_report.json'), 'w') as f:
            json.dump({}, f)
        d = discover_inputs(in_dir, out_dir, SPECS)
        assert [c.rel_path for c in d.candidates] == ['a.png']
        assert any(s.rel_path == 'batch_report.json' for s in d.skipped)


class TestSymlinks:

    def test_link_outside_input_skipped(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        ext_dir = os.path.join(tmpdir_path, 'external')
        os.makedirs(in_dir)
        os.makedirs(ext_dir)
        _png(os.path.join(ext_dir, 'outside.png'))
        os.symlink(os.path.join(ext_dir, 'outside.png'),
                   os.path.join(in_dir, 'link.png'))
        _png(os.path.join(in_dir, 'inside.png'))
        d = discover_inputs(in_dir, os.path.join(tmpdir_path, 'out'), SPECS)
        assert [c.rel_path for c in d.candidates] == ['inside.png']
        skip = {s.rel_path: s for s in d.skipped}['link.png']
        assert 'outside' in skip.reason

    def test_symlink_loop_handled(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        os.symlink('loop.png', os.path.join(in_dir, 'loop.png'))
        _png(os.path.join(in_dir, 'real.png'))
        d = discover_inputs(in_dir, os.path.join(tmpdir_path, 'out'), SPECS)
        assert [c.rel_path for c in d.candidates] == ['real.png']
        skip = {s.rel_path: s for s in d.skipped}['loop.png']
        assert 'loop' in skip.reason

    def test_two_level_loop_handled(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        os.symlink('b.png', os.path.join(in_dir, 'a.png'))
        os.symlink('a.png', os.path.join(in_dir, 'b.png'))
        d = discover_inputs(in_dir, os.path.join(tmpdir_path, 'out'), SPECS)
        assert d.candidates == []
        assert len(d.skipped) == 2
        assert all('loop' in s.reason for s in d.skipped)

    def test_broken_symlink_handled(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        os.symlink('/nonexistent/target.png', os.path.join(in_dir, 'dead.png'))
        d = discover_inputs(in_dir, os.path.join(tmpdir_path, 'out'), SPECS)
        assert d.candidates == []
        assert 'dead.png' == d.skipped[0].rel_path
        assert 'broken' in d.skipped[0].reason

    def test_hardlink_duplicate_collapsed(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'real.png'))
        os.link(os.path.join(in_dir, 'real.png'),
                os.path.join(in_dir, 'alias.png'))
        d = discover_inputs(in_dir, os.path.join(tmpdir_path, 'out'), SPECS)
        assert len(d.candidates) == 1
        assert {c.rel_path for c in d.candidates} <= {'real.png', 'alias.png'}
        assert len(d.skipped) == 1
        assert 'same file' in d.skipped[0].reason

    def test_symlink_to_same_file_collapsed(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        os.makedirs(in_dir)
        _png(os.path.join(in_dir, 'real.png'))
        os.symlink('real.png', os.path.join(in_dir, 'alias.png'))
        d = discover_inputs(in_dir, os.path.join(tmpdir_path, 'out'), SPECS)
        assert sorted(c.rel_path for c in d.candidates) == ['real.png']
        skip = {s.rel_path: s for s in d.skipped}['alias.png']
        assert 'same file' in skip.reason

    def test_internal_symlink_to_directory_skipped(self, tmpdir_path):
        in_dir = os.path.join(tmpdir_path, 'in')
        sub = os.path.join(in_dir, 'sub')
        os.makedirs(sub)
        _png(os.path.join(sub, 'deep.png'))
        os.symlink('sub', os.path.join(in_dir, 'linkdir.png'))
        d = discover_inputs(in_dir, os.path.join(tmpdir_path, 'out'), SPECS)
        assert d.candidates == []
        assert 'not a regular file' in d.skipped[0].reason

    def test_link_to_generated_output_excluded(self, work):
        in_dir, out_dir = work
        _png(os.path.join(in_dir, 'a.png'))
        os.makedirs(out_dir)
        _png(os.path.join(out_dir, 'a_processed.png'))
        # A link inside input pointing at the file this run will regenerate.
        os.symlink(os.path.join(out_dir, 'a_processed.png'),
                   os.path.join(in_dir, 'sneaky.png'))
        d = discover_inputs(in_dir, out_dir, SPECS)
        rels = sorted(c.rel_path for c in d.candidates)
        assert rels == ['a.png']
        assert any('generated by this pipeline' in s.reason
                   for s in d.skipped if s.rel_path == 'sneaky.png')


class TestRaceSafety:

    def test_modified_between_discovery_and_execution(self, work, monkeypatch):
        in_dir, out_dir = work
        p = os.path.join(in_dir, 'a.png')
        _png(p)
        batch = BatchExecutor(_make_pipeline(), in_dir, out_dir, 'cfg')
        discovery = batch.discover()
        assert len(discovery.candidates) == 1
        # Tamper with the file after the set was frozen.
        time.sleep(0.01)
        write_image(alg.generate_checkerboard(10, 10, 2), p, fmt='PNG')
        os.utime(p, None)
        report = batch.run(discovery)
        assert report.succeeded == 0
        assert report.failed == 1
        assert 'modified between discovery and execution' in report.results[0].error

    def test_replaced_by_symlink_after_discovery(self, work, tmpdir_path):
        in_dir, out_dir = work
        p = os.path.join(in_dir, 'a.png')
        _png(p)
        batch = BatchExecutor(_make_pipeline(), in_dir, out_dir, 'cfg')
        discovery = batch.discover()
        other = os.path.join(tmpdir_path, 'other.png')
        _png(other, size=6)
        os.remove(p)
        os.symlink(other, p)
        report = batch.run(discovery)
        assert report.failed == 1
        assert 'changed between discovery and execution' in report.results[0].error

    def test_deleted_between_discovery_and_execution(self, work):
        in_dir, out_dir = work
        p = os.path.join(in_dir, 'a.png')
        _png(p)
        batch = BatchExecutor(_make_pipeline(), in_dir, out_dir, 'cfg')
        discovery = batch.discover()
        os.remove(p)
        report = batch.run(discovery)
        assert report.failed == 1
        assert 'removed between discovery and execution' in report.results[0].error


class TestPlannedNames:

    def test_suffix_and_format(self):
        names = planned_output_names('photo.jpg',
                                     [{'suffix': '_x', 'format': 'PNG'}])
        assert names == {'photo_x.png'}

    def test_multiple_output_nodes(self):
        specs = [{'suffix': '_a', 'format': None},
                 {'suffix': '_b', 'format': 'JPEG'}]
        names = planned_output_names('p.png', specs)
        assert names == {'p_a.png', 'p_b.jpg'}


class TestIsUnder:

    def test_basic(self):
        assert is_under('/a/b', '/a')
        assert is_under('/a', '/a')
        assert not is_under('/ab', '/a')
        assert not is_under('/b', '/a')
