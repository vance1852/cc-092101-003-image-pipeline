"""Frozen candidate discovery for a batch run.

The candidate set is computed once, before anything is written, and is
snapshotted (path, resolved real path, stat fingerprint) so that the execution
loop never re-scans the filesystem. This guarantees that:

* outputs of this very run (including a previous run left in an output
  directory that doubles as the input directory) are excluded up front;
* input/output overlap is decided on normalized *real* paths, so symlinked
  directories are taken into account;
* every filesystem entry that does not become a candidate is reported with a
  precise reason;
* a file changed between discovery and processing is failed loudly instead of
  being silently replaced by whatever version is on disk then.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple
import os
import stat as stat_module

from ..utils.types import SkippedItem, ValidationError
from ..utils.image_io import SUPPORTED_EXTENSIONS, FORMAT_TO_EXT, predict_output_filename

# Entry is a directory (or another non-regular, non-symlink filesystem object).
REASON_DIRECTORY = 'directory'
REASON_UNSUPPORTED_EXTENSION = 'unsupported_extension'
REASON_BATCH_REPORT = 'batch_report'
REASON_STAT_ERROR = 'stat_error'
REASON_BROKEN_SYMLINK = 'broken_symlink'
REASON_SYMLINK_LOOP = 'symlink_loop'
REASON_SYMLINK_NOT_FILE = 'symlink_target_not_file'
REASON_SYMLINK_OUTSIDE = 'symlink_outside_input'
REASON_DUPLICATE_ENTITY = 'duplicate_entity'
REASON_PLANNED_OUTPUT = 'planned_output'
# A file matching the pipeline's output naming signature (suffix + output
# extension) left behind by a previous run, with no surviving source file.
REASON_PIPELINE_PRODUCT = 'pipeline_product'

# Default name of the JSON batch report written into the output directory.
DEFAULT_REPORT_NAME = 'batch_report.json'


@dataclass(frozen=True)
class Candidate:
    # Lexical absolute path used to open the file.
    path: str
    # Fully resolved real path (symlinks followed, normalized).
    real_path: str
    # Path relative to the input directory, for receipts/reports.
    relative_path: str
    size: int
    mtime_ns: int
    dev: int
    ino: int


@dataclass
class DiscoveryResult:
    input_dir: str
    input_real: str
    output_dir: str
    output_real: str
    overlap: bool
    candidates: List[Candidate] = field(default_factory=list)
    skipped: List[SkippedItem] = field(default_factory=list)
    # Real paths of every file this batch is predicted to write.
    planned_outputs: Set[str] = field(default_factory=set)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


def resolve_output_realpath(output_dir: str) -> str:
    """Resolve an output directory that may not exist yet.

    The nearest existing ancestor is resolved with ``realpath`` (honouring any
    symlinks on it); the not-yet-existing tail is appended lexically.
    """
    cur = os.path.abspath(output_dir)
    tail: List[str] = []
    while cur and not os.path.lexists(cur):
        tail.append(os.path.basename(cur))
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    base = os.path.realpath(cur) if cur else os.sep
    if tail:
        return os.path.normpath(os.path.join(base, *reversed(tail)))
    return os.path.normpath(base)


def is_within(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` itself or located below it."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def resolve_symlink_chain(path: str, max_hops: int = 40) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a symlink chain manually so loops can be named explicitly.

    Returns ``(real_path, None)`` or ``(None, reason)`` where reason is one of
    ``REASON_SYMLINK_LOOP`` / ``REASON_BROKEN_SYMLINK``.
    """
    cur = os.path.abspath(path)
    seen: Set[Tuple[int, int]] = set()
    hops = 0
    while os.path.islink(cur):
        try:
            lst = os.lstat(cur)
        except OSError:
            return (None, REASON_BROKEN_SYMLINK)
        key = (lst.st_dev, lst.st_ino)
        if key in seen:
            return (None, REASON_SYMLINK_LOOP)
        seen.add(key)
        hops += 1
        if hops > max_hops:
            return (None, REASON_SYMLINK_LOOP)
        target = os.readlink(cur)
        if not os.path.isabs(target):
            target = os.path.normpath(os.path.join(os.path.dirname(cur), target))
        cur = target
        if not os.path.lexists(cur):
            return (None, REASON_BROKEN_SYMLINK)
    return (os.path.normpath(cur), None)


def make_output_predictor(output_nodes) -> Callable[[str], Set[str]]:
    """Build a function mapping an input filename to predicted output names.

    Mirrors :class:`OutputNode` naming exactly (shared helper in image_io).
    """
    resolved = list(output_nodes)

    def predict(input_filename: str) -> Set[str]:
        names: Set[str] = set()
        for node in resolved:
            params = node.effective_params()
            names.add(predict_output_filename(
                input_filename,
                suffix=params.get('suffix', '') or '',
                fmt=params.get('format'),
            ))
        return names

    return predict


def make_output_signature(output_nodes) -> Callable[[str], bool]:
    """Build a predicate recognising files a previous run likely produced.

    Only output nodes with a *non-empty* suffix are considered: the suffix is
    the pipeline's distinctive marker on generated names. (Output nodes with
    an empty suffix overwrite the source name and are handled exactly by the
    planned-output set, so no heuristic is needed for them.)
    """
    rules = []
    for node in output_nodes:
        params = node.effective_params()
        suffix = params.get('suffix', '') or ''
        if not suffix:
            continue
        fmt = params.get('format')
        fixed_ext = FORMAT_TO_EXT.get(str(fmt).upper()) if fmt else None
        rules.append((suffix, fixed_ext))

    def matches(filename: str) -> bool:
        stem, ext = os.path.splitext(filename)
        ext = ext.lower()
        for suffix, fixed_ext in rules:
            if len(stem) <= len(suffix) or not stem.endswith(suffix):
                continue
            if fixed_ext is not None:
                if ext == fixed_ext:
                    return True
            elif ext in SUPPORTED_EXTENSIONS:
                return True
        return False

    return matches


def discover_candidates(
    input_dir: str,
    output_dir: str,
    predict_outputs: Optional[Callable[[str], Set[str]]] = None,
    recognize_product: Optional[Callable[[str], bool]] = None,
    report_name: str = DEFAULT_REPORT_NAME,
) -> DiscoveryResult:
    """Freeze the candidate set for one batch run.

    ``predict_outputs`` maps an input filename to the set of output filenames
    the pipeline will write. When ``None`` no pipeline-produced files are
    predicted (pipelines without output nodes never write anything).

    ``recognize_product`` optionally recognises filenames a *previous* run left
    behind (by its output suffix/extension signature). It is consulted only
    when input and output overlap, because that is the only situation in which
    a generated file can be re-discovered as input; with separate directories
    normal behavior is untouched.
    """
    input_abs = os.path.abspath(input_dir)
    if not os.path.isdir(input_abs):
        raise ValidationError(f"Input directory does not exist: '{input_dir}'")

    input_real = os.path.normpath(os.path.realpath(input_abs))
    output_abs = os.path.abspath(output_dir)
    output_real = resolve_output_realpath(output_abs)
    overlap = is_within(output_real, input_real) or is_within(input_real, output_real)

    result = DiscoveryResult(
        input_dir=input_abs,
        input_real=input_real,
        output_dir=output_abs,
        output_real=output_real,
        overlap=overlap,
    )
    if predict_outputs is None:
        def predict_outputs(_name: str) -> Set[str]:
            return set()

    # Two-pass discovery:
    #   pass 1 decides which entries are admissible inputs (files/links only,
    #          supported type, inside the input tree, unique entity);
    #   pass 2 computes the planned-output set from the admissible names and
    #          excludes entries that this very run would (re)generate.
    @dataclass
    class _Entry:
        abspath: str
        relpath: str
        name: str
        is_symlink: bool
        real_path: str = ''
        st: Optional[os.stat_result] = None

    admissible: List[_Entry] = []

    def _skip(relpath: str, reason: str, message: str, real_path: Optional[str] = None) -> None:
        result.skipped.append(SkippedItem(relpath, reason, message, real_path))

    try:
        names = sorted(os.listdir(input_abs))
    except OSError as e:
        raise ValidationError(f"Failed to read input directory '{input_abs}': {e}")

    for name in names:
        abspath = os.path.join(input_abs, name)
        relpath = os.path.relpath(abspath, input_abs)
        try:
            lst = os.lstat(abspath)
        except OSError as e:
            _skip(relpath, REASON_STAT_ERROR, f'cannot stat entry: {e}')
            continue
        entry = _Entry(abspath=abspath, relpath=relpath, name=name,
                       is_symlink=stat_module.S_ISLNK(lst.st_mode))

        if name == report_name:
            _skip(relpath, REASON_BATCH_REPORT,
                  'batch report file produced by this pipeline, never an input')
            continue

        if entry.is_symlink:
            target, loop_reason = resolve_symlink_chain(abspath)
            link_note = f"symlink '{relpath}'"
            if loop_reason == REASON_SYMLINK_LOOP:
                _skip(relpath, REASON_SYMLINK_LOOP,
                      f'{link_note} forms a symbolic link loop; skipped')
                continue
            if loop_reason == REASON_BROKEN_SYMLINK:
                _skip(relpath, REASON_BROKEN_SYMLINK,
                      f"{link_note} points to a target that does not exist; skipped")
                continue
            try:
                tst = os.stat(target)
            except OSError as e:
                _skip(relpath, REASON_BROKEN_SYMLINK,
                      f'{link_note} target cannot be accessed: {e}', target)
                continue
            if not stat_module.S_ISREG(tst.st_mode):
                kind = 'directory' if stat_module.S_ISDIR(tst.st_mode) else 'non-regular file'
                _skip(relpath, REASON_SYMLINK_NOT_FILE,
                      f'{link_note} resolves to a {kind} outside the file scan: {target}',
                      target)
                continue
            entry.real_path = os.path.normpath(target)
            entry.st = tst
        else:
            if stat_module.S_ISDIR(lst.st_mode):
                _skip(relpath, REASON_DIRECTORY, 'subdirectory; image scan is non-recursive')
                continue
            if not stat_module.S_ISREG(lst.st_mode):
                _skip(relpath, REASON_DIRECTORY, 'not a regular file; skipped')
                continue
            entry.real_path = os.path.normpath(os.path.realpath(abspath))
            entry.st = lst

        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            _skip(relpath, REASON_UNSUPPORTED_EXTENSION,
                  f"extension '{ext or '(none)'}' is not a supported image format")
            continue

        if entry.is_symlink and not is_within(entry.real_path, input_real):
            _skip(relpath, REASON_SYMLINK_OUTSIDE,
                  f"symlink points outside the input directory: {entry.real_path}",
                  entry.real_path)
            continue

        admissible.append(entry)

    # One entity reached by several names (hard links or symlinks to the same
    # file) is processed exactly once. Prefer a non-symlink name, then the
    # shortest relative path, then lexical order, so the real name wins over an
    # alias.
    entity_groups: Dict[Tuple[int, int], List[_Entry]] = {}
    for entry in admissible:
        entity_groups.setdefault((entry.st.st_dev, entry.st.st_ino), []).append(entry)

    chosen_by_entity: Dict[Tuple[int, int], _Entry] = {}
    for entity, entries in entity_groups.items():
        entries.sort(key=lambda e: (e.is_symlink, len(e.relpath), e.relpath))
        chosen_by_entity[entity] = entries[0]

    # Pass 2: everything this run is predicted to write, as real paths. Output
    # names are derived from the chosen name of each entity (the filename the
    # pipeline actually receives).
    planned: Set[str] = set()
    for entry in chosen_by_entity.values():
        for out_name in predict_outputs(entry.name):
            planned.add(os.path.normpath(os.path.join(output_real, out_name)))
    result.planned_outputs = planned

    # Final decision per entity, applied in scan order for stable receipts.
    for entry in admissible:
        entity = (entry.st.st_dev, entry.st.st_ino)
        chosen = chosen_by_entity[entity]
        if chosen.real_path in planned:
            if entry is chosen:
                _skip(entry.relpath, REASON_PLANNED_OUTPUT,
                      f"file is an output this pipeline generates ({entry.name}); "
                      f"excluded to keep repeated runs from re-processing their own products",
                      entry.real_path)
            else:
                _skip(entry.relpath, REASON_PLANNED_OUTPUT,
                      f"same file as '{chosen.relpath}', which is an output this pipeline "
                      f"generates; excluded",
                      entry.real_path)
            continue
        # Orphan product from a previous run (its source no longer exists).
        # Only applied when input/output overlap and the name carries the
        # pipeline's output signature; separate directories are unaffected.
        is_orphan_product = (
            overlap
            and recognize_product is not None
            and recognize_product(chosen.name)
        )
        if is_orphan_product:
            if entry is chosen:
                _skip(entry.relpath, REASON_PIPELINE_PRODUCT,
                      f"filename '{chosen.name}' matches an output this pipeline generates "
                      f"(left over from a previous run); excluded to prevent suffix stacking. "
                      f"If this is a real input, rename it and/or use a separate output directory",
                      chosen.real_path)
            else:
                _skip(entry.relpath, REASON_PIPELINE_PRODUCT,
                      f"same file as '{chosen.relpath}', a leftover product from a previous "
                      f"run; excluded",
                      entry.real_path)
            continue
        if entry is not chosen:
            _skip(entry.relpath, REASON_DUPLICATE_ENTITY,
                  f"same file (device {entry.st.st_dev}, inode {entry.st.st_ino}) already "
                  f"scheduled as '{chosen.relpath}'; hard/soft links to one entity are "
                  f"processed once",
                  entry.real_path)
            continue
        result.candidates.append(Candidate(
            path=entry.abspath,
            real_path=entry.real_path,
            relative_path=entry.relpath,
            size=entry.st.st_size,
            mtime_ns=entry.st.st_mtime_ns,
            dev=entry.st.st_dev,
            ino=entry.st.st_ino,
        ))
    return result


def verify_candidate(candidate: Candidate, planned_outputs: Set[str]) -> Optional[str]:
    """Re-check a frozen candidate immediately before processing it.

    Returns ``None`` when the on-disk file is still exactly the one discovered,
    otherwise a human-readable explanation of what changed.
    """
    try:
        st = os.stat(candidate.path)
    except FileNotFoundError:
        return 'file removed after candidate set was frozen; not processing a substitute'
    except OSError as e:
        return f'file cannot be accessed after candidate set was frozen: {e}'
    if not stat_module.S_ISREG(st.st_mode):
        return 'path is no longer a regular file; refusing to process the replacement'
    if (st.st_dev, st.st_ino) != (candidate.dev, candidate.ino):
        return 'file was replaced (inode changed) after candidate set was frozen; refusing to process a different version'
    if st.st_size != candidate.size or st.st_mtime_ns != candidate.mtime_ns:
        return 'file content changed (size/mtime differ) after candidate set was frozen; refusing to process a different version'
    real_now = os.path.normpath(os.path.realpath(candidate.path))
    if real_now != candidate.real_path:
        return f'symlink target changed (was {candidate.real_path}, now {real_now}); refusing to follow the new target'
    if real_now in planned_outputs:
        return 'path now resolves to a file this run would write; refusing to process its own output'
    return None
