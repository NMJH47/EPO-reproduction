"""Import-safe fixed rollout boundary insertion; runtime CUDA is supplied explicitly."""
import ast
import copy
import ctypes
import difflib
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import textwrap
import time
import types

POLICIES = ('no_cache', 'step_cache')
BOUNDARIES_PER_EPOCH = 17
MIN_DRIVER_FREE = 1024 ** 3
MIN_DISK_FREE = 1024 ** 3


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def transform_source(source):
    """Only add entry and loop-tail callbacks; refuse ambiguous loop structure."""
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), 'One plain function required')
    function = tree.body[0]
    require(function.name == 'play_steps' and not function.decorator_list, 'Expected undecorated play_steps')
    require([arg.arg for arg in function.args.args] == ['self'] and not function.args.vararg
            and not function.args.kwarg and not function.args.kwonlyargs, 'Unexpected play_steps signature')
    loops = [node for node in function.body if isinstance(node, ast.For)
             and ast.dump(node.iter) == ast.dump(ast.parse('range(self.horizon_length)', mode='eval').body)]
    require(len(loops) == 1, 'Exactly one top-level horizon loop required')
    loop = loops[0]
    require(isinstance(loop.target, ast.Name) and loop.target.id == 'n' and not loop.orelse, 'Unexpected loop index/else')
    require(not any(isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Yield, ast.YieldFrom))
                    for node in ast.walk(loop)), 'Early loop exit would skip a completed boundary')
    last = ast.parse('self.current_lengths = self.current_lengths * not_dones').body[0]
    require(ast.dump(loop.body[-1]) == ast.dump(last), 'Expected final reward/length bookkeeping assignment')
    require(not any(isinstance(node, ast.Attribute) and node.attr == '_cache_screen_boundary'
                    for node in ast.walk(function)), 'Boundary already installed')
    original = copy.deepcopy(tree)
    entry = ast.parse("self._cache_screen_boundary('entry', -1)").body[0]
    tail = ast.parse("self._cache_screen_boundary('step_end', n)").body[0]
    function.body.insert(0, entry)
    loop.body.append(tail)
    ast.fix_missing_locations(tree)
    stripped = copy.deepcopy(tree)
    stripped.body[0].body.pop(0)
    stripped_loop = [node for node in stripped.body[0].body if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name) and node.target.id == 'n'][0]
    stripped_loop.body.pop()
    require(ast.dump(stripped) == ast.dump(original), 'Removal of exactly two callbacks must reproduce the full original AST')
    # Python3.8 has no ast.unparse; preserve original lines for the reviewable patch.
    lines = source.splitlines(keepends=True)
    at_entry = original.body[0].body[0].lineno - 1
    at_tail = loops[0].body[-2].end_lineno
    first_indent = ' ' * original.body[0].body[0].col_offset
    tail_indent = ' ' * loops[0].body[-2].col_offset
    lines.insert(at_tail, tail_indent + "self._cache_screen_boundary('step_end', n)\n")
    lines.insert(at_entry, first_indent + "self._cache_screen_boundary('entry', -1)\n")
    patched_text = ''.join(lines)
    require(ast.dump(ast.parse(patched_text)) == ast.dump(tree), 'Readable patch and compiled AST differ')
    return tree, patched_text, {'original_function_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'patched_function_sha256': hashlib.sha256(patched_text.encode()).hexdigest(),
        'exact_AST_after_removing_two_callbacks': True,
        'insertion_points': ['before original first statement', 'after current_lengths reset at horizon loop tail']}


def compile_play_steps(original, output):
    require(not original.__closure__, 'Captured closure unsupported')
    source = textwrap.dedent(inspect.getsource(original))
    tree, patched, proof = transform_source(source)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'play_steps.before.py').write_text(source)
    (output / 'play_steps.after.py').write_text(patched)
    (output / 'play_steps.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True),
                                          fromfile='original/play_steps.py', tofile='screen/play_steps.py')))
    write(output / 'play_steps_AST_proof.json', proof)
    namespace = dict(original.__globals__)
    exec(compile(tree, str(output / 'play_steps.after.py'), 'exec'), namespace)
    replacement = namespace['play_steps']
    replacement.__defaults__, replacement.__kwdefaults__ = original.__defaults__, original.__kwdefaults__
    return replacement, proof


class NvmlMemory:
    """Read driver-wide GPU0 counters; no tensors, forwards or random draws."""
    class Memory(ctypes.Structure):
        _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong), ('used', ctypes.c_ulonglong)]

    def __init__(self):
        self.lib = ctypes.CDLL('libnvidia-ml.so.1')
        self.handle = ctypes.c_void_p()
        self.lib.nvmlInit_v2.restype = ctypes.c_int
        self.lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Memory)]
        require(self.lib.nvmlInit_v2() == 0, 'NVML init failed')
        require(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) == 0, 'NVML GPU0 unavailable')

    def __call__(self):
        memory = self.Memory()
        require(self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(memory)) == 0, 'NVML memory query failed')
        return {'driver_total_bytes': memory.total, 'driver_free_bytes': memory.free, 'driver_used_bytes': memory.used}


class CacheBoundaryObserver:
    def __init__(self, agent, policy, output, torch_runtime, driver, disk_path,
                 expected_epochs=20, allow_cpu_fixture=False):
        require(policy in POLICIES, 'Unknown resource policy')
        require(allow_cpu_fixture or torch_runtime.cuda.is_available(), 'CUDA-only runtime')
        require(agent.horizon_length == 16 and agent.seq_length == 16 and agent.num_actors == 20480
                and agent.is_rnn and not agent.multi_gpu and not agent.has_central_value,
                'Only actual E51 single-GPU recurrent layout is supported')
        self.agent, self.policy, self.output = agent, policy, Path(output)
        self.torch, self.driver, self.disk_path = torch_runtime, driver, Path(disk_path)
        self.expected_epochs = expected_epochs
        self.rows, self.epochs = [], []
        self.active_epoch, self.next_step = None, None
        self.output.mkdir(parents=True, exist_ok=True)
        self.original_epoch = agent.train_epoch
        self.had_boundary = hasattr(agent, '_cache_screen_boundary')
        require(not self.had_boundary, 'Refusing an existing boundary observer')
        agent._cache_screen_boundary = self.boundary

        def train_epoch(unused_self):
            try:
                result = self.original_epoch()
                rows = [row for row in self.rows if row['epoch'] == int(agent.epoch_num)]
                require(len(rows) == BOUNDARIES_PER_EPOCH and self.next_step == 16, 'Exact17 callbacks missing in epoch')
                play, update, core = map(float, result[1:4])
                require(abs(play + update - core) < 1e-5 and min(play, update, core) > 0, 'Invalid actual core timing')
                self.epochs.append({'epoch': int(agent.epoch_num), 'frame_before_increment': int(agent.frame),
                    'play_seconds': play, 'update_seconds': update, 'core_seconds': core,
                    'physical_frames': 327680, 'boundary_count': len(rows),
                    'cache_release_calls': sum(row['cache_release_called'] for row in rows)})
                self.persist('running')
                return result
            except BaseException as exc:
                self.persist('failed', repr(exc))
                raise

        agent.train_epoch = types.MethodType(train_epoch, agent)

    def boundary(self, kind, index):
        epoch = int(self.agent.epoch_num)
        require(kind in ('entry', 'step_end'), 'Unknown boundary')
        if kind == 'entry':
            require(index == -1 and (self.active_epoch is None or epoch > self.active_epoch)
                    and (self.next_step is None or self.next_step == 16), 'Repeated/out-of-order rollout entry')
            self.active_epoch, self.next_step = epoch, 0
        else:
            require(epoch == self.active_epoch and index == self.next_step and 0 <= index < 16,
                    'Missing/repeated/out-of-order complete step boundary')
            self.next_step += 1
        cuda = self.torch.cuda
        row = {'epoch': epoch, 'frame_before_increment': int(self.agent.frame), 'boundary': kind, 'step_index': index,
               'policy': self.policy, 'allocated_before': int(cuda.memory_allocated()),
               'reserved_before': int(cuda.memory_reserved()), **{'before_' + k: v for k, v in self.driver().items()}}
        started = time.perf_counter()
        try:
            if self.policy == 'step_cache':
                cuda.empty_cache()
            row.update(cache_release_called=self.policy == 'step_cache', boundary_action_seconds=time.perf_counter()-started,
                       allocated_after=int(cuda.memory_allocated()), reserved_after=int(cuda.memory_reserved()),
                       **{'after_' + k: v for k, v in self.driver().items()})
            row['disk_free_bytes'] = shutil.disk_usage(self.disk_path).free
            self.rows.append(row)
            require(row['after_driver_free_bytes'] >= MIN_DRIVER_FREE, 'Post-boundary driver free below1GiB')
            require(row['disk_free_bytes'] >= MIN_DISK_FREE, 'Disk below1GiB floor')
        except BaseException as exc:
            if not self.rows or self.rows[-1] is not row:
                row.update(error=repr(exc));self.rows.append(row)
            self.persist('failed', repr(exc))
            raise

    def persist(self, status, error=None):
        write(self.output / 'boundaries.json', self.rows)
        write(self.output / 'epochs.json', self.epochs)
        write(self.output / 'state.json', {'status': status, 'policy': self.policy, 'observed_epochs': len(self.epochs),
            'observed_boundaries': len(self.rows), 'expected_boundaries': self.expected_epochs * 17,
            'cache_release_calls': sum(row.get('cache_release_called', False) for row in self.rows), 'error': error})

    def finalize(self):
        try:
            require(len(self.epochs) == self.expected_epochs and len(self.rows) == self.expected_epochs * 17,
                    'Incomplete fixed epoch/boundary budget')
            require([row['epoch'] for row in self.epochs] == list(range(15260, 15260+self.expected_epochs)), 'Epoch coverage differs')
            require(all(row['frame_before_increment'] == 5000069120+i*327680 for i,row in enumerate(self.epochs)), 'Frame coverage differs')
            self.persist('completed')
            return json.loads((self.output / 'state.json').read_text())
        except BaseException as exc:
            self.persist('failed', repr(exc));raise

    def uninstall(self):
        self.agent.train_epoch = self.original_epoch
        if hasattr(self.agent, '_cache_screen_boundary'):
            del self.agent._cache_screen_boundary
