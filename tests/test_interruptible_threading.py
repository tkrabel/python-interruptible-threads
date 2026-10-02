"""Tests for interruptible_threading.

Patches mutate global stdlib state, so each test installs/uninstalls via the
``patched`` fixture (function-scoped, try/finally) to stay isolated.
"""
from __future__ import annotations

import contextlib
import os
import queue
import select
import selectors
import socket
import sys
import threading
import time

import pytest

import interruptible_threading as it
from interruptible_threading import (
    InterruptibleThread,
    ThreadInterrupted,
    check_interrupt,
    clear_interrupt,
    critical_section,
    interruptible_recv,
    is_interrupted,
)

pytestmark = pytest.mark.timeout(30)

# Real (unpatched) sleep for use by the main test thread while patches are live.
_REAL_SLEEP = time.sleep

# Async injection (PyThreadState_SetAsyncExc) is disabled while a trace hook is
# active (e.g. coverage's C tracer on <=3.11), since it can deadlock the target.
# Tests that can only be delivered by async injection -- a pure-Python loop with
# no checkpoint -- are skipped in that case. They still run under plain pytest
# (no coverage) and under coverage's sys.monitoring core (3.12+).
needs_async_injection = pytest.mark.skipif(
    it._tracing_active(),
    reason="async injection disabled under an active trace hook (coverage C tracer)",
)


@pytest.fixture
def patched():
    InterruptibleThread.install_patches()
    try:
        yield
    finally:
        InterruptibleThread.uninstall_patches()


@pytest.fixture(autouse=True)
def _no_leaked_workers():
    """Ensure no worker thread outlives its test.

    Most tests only ``res.done.wait(...)`` and never ``t.join()``, so a worker can
    still be unwinding ``InterruptibleThread.run()``'s cleanup when the test ends.
    A daemon worker that is still alive at pytest session finish can deadlock
    pytest-cov's coverage-data save (collector lock vs. a live traced thread) --
    which is exactly how a CI job once hung for 6h. Interrupt and join any
    stragglers here, and fail loudly if one cannot be joined.
    """
    yield
    stragglers = [
        t
        for t in threading.enumerate()
        if isinstance(t, InterruptibleThread) and t.is_alive()
    ]
    for t in stragglers:
        try:
            t.interrupt(recursive=True)
        except (ValueError, RuntimeError):
            pass
    for t in stragglers:
        t.join(5)
    leaked = [t for t in stragglers if t.is_alive()]
    assert not leaked, f"worker(s) outlived the test (possible hang): {leaked}"


class Result:
    """Captures a worker's outcome and exposes lifecycle events."""

    def __init__(self) -> None:
        self.exc: BaseException | None = None
        self.value = None
        self.started = threading.Event()
        self.done = threading.Event()


def run_worker(
    fn, *, daemon=True, thread_cls=InterruptibleThread
) -> tuple[InterruptibleThread, Result]:
    res = Result()

    def target():
        res.started.set()
        try:
            res.value = fn(res)
        except BaseException as e:  # noqa: BLE001 - we want to capture everything
            res.exc = e
        finally:
            res.done.set()

    t = thread_cls(target=target, daemon=daemon)
    t.start()
    res.started.wait(5)
    return t, res


def spin(seconds: float) -> None:
    """Busy-wait (so async injection has bytecode boundaries to fire on)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


# --------------------------------------------------------------------------- #


@needs_async_injection
def test_async_injection_breaks_pure_python_loop(patched):
    def fn(res):
        while True:
            pass

    t, res = run_worker(fn)
    _REAL_SLEEP(0.05)
    t.interrupt()
    assert res.done.wait(2), "worker did not stop"
    assert isinstance(res.exc, ThreadInterrupted)


def test_sleep_interrupted_promptly(patched):
    def fn(res):
        time.sleep(100)

    t, res = run_worker(fn)
    _REAL_SLEEP(0.05)
    start = time.monotonic()
    t.interrupt()
    assert res.done.wait(2)
    assert isinstance(res.exc, ThreadInterrupted)
    assert time.monotonic() - start < 1.0


def test_sleep_interrupt_race_stress(patched):
    import random

    for _ in range(150):

        def fn(res):
            time.sleep(5)

        t, res = run_worker(fn)
        spin(random.uniform(0.0, 0.002))
        t.interrupt()
        assert res.done.wait(2), "worker hung on interrupt"
        assert isinstance(res.exc, ThreadInterrupted)
        t.join(2)
        assert not t.is_alive()


@contextlib.contextmanager
def _global_tracing():
    """Ensure a trace hook is active globally (main + every spawned thread), as
    coverage's C tracer is, so ``_tracing_active()`` is True and ``interrupt()``
    must deliver cooperatively (no async injection). If already tracing (running
    under coverage), do nothing; otherwise install a no-op tracer and restore.
    """
    if it._tracing_active():
        yield
        return

    def _tracer(frame, event, arg):
        return _tracer

    old_sys = sys.gettrace()
    sys.settrace(_tracer)
    threading.settrace(_tracer)
    try:
        assert it._tracing_active()
        yield
    finally:
        # Nothing was installed before (we checked), so restoring to None is safe.
        threading.settrace(None)  # type: ignore[arg-type]
        sys.settrace(old_sys)


def test_interrupt_not_lost_entering_sleep_under_tracing(patched):
    # Regression for the lost-wakeup race that hung CI: interrupt() fires while
    # the worker has decided to block but has not yet committed to the primitive
    # (the sleeping/selecting hints still read False). Under tracing, async
    # injection is disabled, so only the durable `pending` flag delivers -- and
    # the zero-delay interrupt lands the worker mid-entry into the sleep.
    with _global_tracing():
        for _ in range(60):

            def fn(res):
                time.sleep(100)

            t, res = run_worker(fn)
            t.interrupt()  # no delay: worker is entering _coop_sleep right now
            assert res.done.wait(3), "interrupt was lost; worker slept through it"
            assert isinstance(res.exc, ThreadInterrupted)
            t.join(2)
            assert not t.is_alive()


def test_interrupt_sleeping_thread_under_tracing_no_deadlock(patched):
    # Regression for the deadlock that hung CI under coverage. Two distinct bugs
    # met here: (1) every cooperative delivery called PyThreadState_SetAsyncExc
    # to clear a possibly-armed async exception, and (2) interrupt() async-injected
    # speculatively. Either SetAsyncExc call wedges a lock-touching thread while a
    # trace hook is active. With a global tracer installed, neither may fire.
    with _global_tracing():
        for _ in range(60):

            def fn(res):
                time.sleep(100)

            t, res = run_worker(fn)
            _REAL_SLEEP(0.02)  # let the worker reach the parked cond.wait
            t.interrupt()
            assert res.done.wait(3), "deadlock: interrupt under tracer never delivered"
            assert isinstance(res.exc, ThreadInterrupted)
            t.join(2)
            assert not t.is_alive()


def test_select_interrupted(patched):
    import select

    r, w = os.pipe()
    try:

        def fn(res):
            select.select([r], [], [])

        t, res = run_worker(fn)
        _REAL_SLEEP(0.05)
        t.interrupt()
        assert res.done.wait(2)
        assert isinstance(res.exc, ThreadInterrupted)
    finally:
        os.close(r)
        os.close(w)


def test_asyncio_clean_cancellation(patched):
    def fn(res):
        import asyncio

        InterruptibleThread.run_interruptible(asyncio.sleep(100))

    t, res = run_worker(fn)
    _REAL_SLEEP(0.1)
    start = time.monotonic()
    t.interrupt()
    assert res.done.wait(3)
    assert isinstance(res.exc, ThreadInterrupted)
    assert time.monotonic() - start < 2.0


def test_recursive_interrupt(patched):
    def parent_fn(res):
        kids, kres = [], []
        for _ in range(2):
            kr = Result()

            def ktarget(kr=kr):
                kr.started.set()
                try:
                    time.sleep(100)
                except BaseException as e:  # noqa: BLE001
                    kr.exc = e
                finally:
                    kr.done.set()

            kt = InterruptibleThread(target=ktarget, daemon=True)
            kt.start()
            kr.started.wait(2)
            kids.append(kt)
            kres.append(kr)
        res.kids = kids
        res.kres = kres
        time.sleep(100)

    t, res = run_worker(parent_fn)
    while not getattr(res, "kres", None):
        _REAL_SLEEP(0.01)
    _REAL_SLEEP(0.1)
    t.interrupt(recursive=True)
    assert res.done.wait(3)
    assert isinstance(res.exc, ThreadInterrupted)
    for kr in res.kres:
        assert kr.done.wait(2)
        assert isinstance(kr.exc, ThreadInterrupted)


def test_child_removed_from_parent_after_exit(patched):
    release = threading.Event()

    def parent_fn(res):
        kr = Result()

        def ktarget():
            kr.started.set()
            # exits immediately

        kt = InterruptibleThread(target=ktarget, daemon=True)
        kt.start()
        kt.join(2)
        res.child_tid = kt.ident
        res.child_done = True
        # Stay alive until released so the parent's _State is still inspectable.
        while not release.is_set():
            spin(0.01)

    t, res = run_worker(parent_fn)
    while not getattr(res, "child_done", False):
        _REAL_SLEEP(0.01)
    _REAL_SLEEP(0.05)
    parent_st = it._State.get_state_by_ident(t.ident)
    assert parent_st is not None
    with parent_st.cancel_cond:
        assert res.child_tid not in parent_st.children
    # The exited child is fully unregistered.
    assert it._State.get_state_by_ident(res.child_tid) is None
    release.set()
    t.join(2)


def test_masking_defers_until_exit(patched):
    in_mask = threading.Event()

    def fn(res):
        with critical_section():
            in_mask.set()
            spin(0.4)
            res.pending_in_mask = is_interrupted()
        res.after_mask = True  # should not be reached

    t, res = run_worker(fn)
    assert in_mask.wait(2)
    t.interrupt()
    assert res.done.wait(3)
    assert isinstance(res.exc, ThreadInterrupted)
    assert res.pending_in_mask is True
    assert not getattr(res, "after_mask", False)


def test_nested_masks_raise_only_at_outermost_exit(patched):
    in_mask = threading.Event()

    def fn(res):
        with critical_section():
            with critical_section():
                in_mask.set()
                spin(0.3)
            res.between = True
            res.between_pending = is_interrupted()
            spin(0.05)
        res.after = True  # should not be reached

    t, res = run_worker(fn)
    assert in_mask.wait(2)
    t.interrupt()
    assert res.done.wait(3)
    assert isinstance(res.exc, ThreadInterrupted)
    assert res.between is True
    assert res.between_pending is True
    assert not getattr(res, "after", False)


def test_check_interrupt_in_cpu_loop(patched):
    def fn(res):
        while True:
            check_interrupt()

    t, res = run_worker(fn)
    _REAL_SLEEP(0.05)
    t.interrupt()
    assert res.done.wait(2)
    assert isinstance(res.exc, ThreadInterrupted)


def test_clear_interrupt(patched):
    # clear_interrupt() is meaningful while masked: the flag is set but no async
    # exception is armed, so the thread can observe and swallow it.
    in_mask = threading.Event()

    def fn(res):
        with critical_section():
            in_mask.set()
            while not is_interrupted():
                spin(0.01)
            res.was_interrupted = clear_interrupt()
        res.still_pending = is_interrupted()
        res.after = True

    t, res = run_worker(fn)
    assert in_mask.wait(2)
    t.interrupt()
    assert res.done.wait(2)
    assert res.was_interrupted is True
    assert res.still_pending is False
    assert getattr(res, "after", False) is True
    # Cleared cleanly, so no exception escaped.
    assert res.exc is None


@needs_async_injection
def test_caught_async_interrupt_not_redelivered(patched):
    def fn(res):
        try:
            while True:
                pass
        except ThreadInterrupted:
            res.caught = True
            # The durable pending flag survives an async-injected delivery (so a
            # thread that parks before the injection fires is never lost); a
            # thread that catches and *continues* clears it explicitly, Java's
            # Thread.interrupted() style. After that, checkpoints must not raise.
            clear_interrupt()
        for _ in range(10000):
            check_interrupt()
        res.no_redeliver = True

    t, res = run_worker(fn)
    _REAL_SLEEP(0.05)
    t.interrupt()
    assert res.done.wait(2)
    assert getattr(res, "caught", False) is True
    assert getattr(res, "no_redeliver", False) is True
    assert res.exc is None


@pytest.mark.parametrize(
    "block",
    [
        pytest.param(lambda: threading.Event().wait(), id="event"),
        pytest.param(lambda: queue.Queue().get(), id="queue"),
        pytest.param(lambda: _cond_wait(), id="condition"),
    ],
)
def test_chunked_primitives_interrupted(patched, block):
    def fn(res):
        block()

    t, res = run_worker(fn)
    _REAL_SLEEP(0.05)
    start = time.monotonic()
    t.interrupt()
    assert res.done.wait(2)
    assert isinstance(res.exc, ThreadInterrupted)
    assert time.monotonic() - start < 1.0


def _cond_wait():
    c = threading.Condition()
    with c:
        c.wait()


def test_interruptible_recv(patched):
    a, b = socket.socketpair()
    prev_timeout = a.gettimeout()
    try:

        def fn(res):
            interruptible_recv(a, 100)

        t, res = run_worker(fn)
        _REAL_SLEEP(0.05)
        t.interrupt()
        assert res.done.wait(2)
        assert isinstance(res.exc, ThreadInterrupted)
        assert a.gettimeout() == prev_timeout
    finally:
        a.close()
        b.close()


@pytest.mark.skipif(not os.path.isdir("/proc/self/fd"), reason="needs /proc")
def test_lazy_pipe_no_fd_growth(patched):
    def count_fds():
        return len(os.listdir("/proc/self/fd"))

    before = count_fds()
    threads = []
    for _ in range(40):

        def fn(res):
            time.sleep(100)  # uses Condition, not the self-pipe

        t, res = run_worker(fn)
        threads.append((t, res))
    after_spawn = count_fds()
    # Sleeping threads must not each allocate a 2-fd self-pipe.
    assert after_spawn - before < 40
    for t, _ in threads:
        t.interrupt()
    for t, res in threads:
        assert res.done.wait(2)
        t.join(2)


def test_legacy_keyboardinterrupt_mode():
    InterruptibleThread.install_patches(legacy_keyboardinterrupt=True)
    try:

        def fn(res):
            try:
                time.sleep(100)
            except KeyboardInterrupt:
                res.caught_ki = True
                raise

        t, res = run_worker(fn)
        _REAL_SLEEP(0.05)
        t.interrupt()
        assert res.done.wait(2)
        assert getattr(res, "caught_ki", False) is True
        assert isinstance(res.exc, KeyboardInterrupt)
        assert isinstance(res.exc, ThreadInterrupted)
    finally:
        InterruptibleThread.uninstall_patches()


def test_interrupt_finished_thread_raises_value_error(patched):
    def fn(res):
        return 42

    t, res = run_worker(fn)
    assert res.done.wait(2)
    t.join(2)
    with pytest.raises(ValueError):
        t.interrupt()


def test_main_thread_checkpoint_is_noop(patched):
    # No interruptible state on the main thread; must not raise.
    check_interrupt()
    assert is_interrupted() is False


def test_patch_round_trip():
    assert threading.Thread is it._ORIG_THREAD
    InterruptibleThread.install_patches()
    try:
        with pytest.raises(ValueError):
            InterruptibleThread.install_patches()
    finally:
        InterruptibleThread.uninstall_patches()
    assert threading.Thread is it._ORIG_THREAD
    assert time.sleep is it._ORIG_SLEEP
    assert threading.Condition.wait is it._ORIG_COND_WAIT


class HostedThread(InterruptibleThread):
    """Model a framework that owns run() and registers only individual commands."""

    def run(self):
        it._ORIG_THREAD.run(self)


@pytest.fixture
def hosted_patches():
    InterruptibleThread.install_patches(
        monkeypatch_thread=False, monkeypatch_selector=False
    )
    try:
        yield
    finally:
        InterruptibleThread.uninstall_patches()


@pytest.mark.parametrize("patch_thread", [False, True])
@pytest.mark.parametrize("patch_selector", [False, True])
def test_optional_patch_round_trip(patch_thread, patch_selector):
    original_thread = threading.Thread
    original_selector = selectors.DefaultSelector
    InterruptibleThread.install_patches(
        monkeypatch_thread=patch_thread, monkeypatch_selector=patch_selector
    )
    try:

        def fn(res):
            return isinstance(threading.Thread(), InterruptibleThread)

        t, res = run_worker(fn)
        t.join(2)
        assert not t.is_alive()
        assert res.exc is None
        assert res.value is patch_thread
        assert (selectors.DefaultSelector is not original_selector) is patch_selector
    finally:
        InterruptibleThread.uninstall_patches()
    assert threading.Thread is original_thread
    assert selectors.DefaultSelector is original_selector
    assert time.sleep is _REAL_SLEEP


@pytest.mark.parametrize("block", ["sleep", "select", "event", "queue"])
def test_scoped_blocking_interrupt_reuses_host_thread(hosted_patches, block):
    ready = queue.Queue()

    def fn(res):
        thread = threading.current_thread()
        interrupted = []
        for command in range(3):
            with thread.interruption_scope(cooperative_only=True):
                ready.put(command)
                try:
                    if block == "sleep":
                        time.sleep(10)
                    elif block == "select":
                        select.select([], [], [], 10)
                    elif block == "event":
                        threading.Event().wait(10)
                    else:
                        queue.Queue().get(timeout=10)
                except ThreadInterrupted:
                    interrupted.append(command)
            # The enclosing host continues outside each execution scope.
            check_interrupt()
            assert not is_interrupted()
        return interrupted

    t, res = run_worker(fn, thread_cls=HostedThread)
    for command in range(3):
        assert ready.get(timeout=2) == command
        t.interrupt()
    t.join(3)
    assert not t.is_alive()
    assert res.exc is None
    assert res.value == [0, 1, 2]


@pytest.mark.parametrize("tracing", [False, True])
def test_cooperative_scope_waits_for_checkpoint(hosted_patches, tracing):
    ready = threading.Event()

    def fn(res):
        with threading.current_thread().interruption_scope(cooperative_only=True):
            ready.set()
            deadline = time.monotonic() + 3
            while not is_interrupted():
                assert time.monotonic() < deadline
            # No async exception may escape into this computation, even without tracing.
            res.before_checkpoint = True
            with pytest.raises(ThreadInterrupted):
                check_interrupt()
            assert not is_interrupted()
            check_interrupt()
        return 42

    with _global_tracing() if tracing else contextlib.nullcontext():
        t, res = run_worker(fn, thread_cls=HostedThread)
        assert ready.wait(2)
        t.interrupt()
        t.join(3)
    assert not t.is_alive()
    assert res.exc is None
    assert res.before_checkpoint
    assert res.value == 42


@pytest.mark.parametrize("masked", [False, True])
def test_scope_predicate_defers_delivery(hosted_patches, masked):
    ready = threading.Event()

    def fn(res):
        allowed = False
        with threading.current_thread().interruption_scope(
            cooperative_only=True, should_interrupt=lambda: allowed
        ):
            with critical_section() if masked else contextlib.nullcontext():
                ready.set()
                deadline = time.monotonic() + 3
                while not is_interrupted():
                    assert time.monotonic() < deadline
                check_interrupt()
                time.sleep(0)
            assert is_interrupted()
            allowed = True
            with pytest.raises(ThreadInterrupted):
                check_interrupt()
            assert not is_interrupted()
        return 42

    t, res = run_worker(fn, thread_cls=HostedThread)
    assert ready.wait(2)
    t.interrupt()
    t.join(3)
    assert not t.is_alive()
    assert res.exc is None
    assert res.value == 42


def test_scope_failure_clears_pending_and_closes_pipe(hosted_patches):
    ready = threading.Event()

    def fn(res):
        thread = threading.current_thread()
        with pytest.raises(
            ValueError, match="command failed"
        ), thread.interruption_scope(
            cooperative_only=True, should_interrupt=lambda: False
        ):
            select.select([], [], [], 0)
            state = it._State.get_state_by_ident()
            fds = (state.rfd, state.wfd)
            ready.set()
            deadline = time.monotonic() + 3
            while not is_interrupted():
                assert time.monotonic() < deadline
            raise ValueError("command failed")
        for fd in fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        with thread.interruption_scope(cooperative_only=True):
            check_interrupt()
            time.sleep(0)
        return 42

    t, res = run_worker(fn, thread_cls=HostedThread)
    assert ready.wait(2)
    t.interrupt()
    t.join(3)
    assert not t.is_alive()
    assert res.exc is None
    assert res.value == 42


def test_cooperative_scope_closed_loop_keeps_interrupt_pending(hosted_patches):
    import asyncio

    def fn(res):
        thread = threading.current_thread()
        with thread.interruption_scope(cooperative_only=True):
            # Exercise the existing fallback for a loop closed before cancellation.
            loop = asyncio.new_event_loop()
            loop.close()
            state = it._State.get_state_by_ident()
            state.event_loop = loop
            try:
                thread.interrupt()
                assert is_interrupted()
                with pytest.raises(ThreadInterrupted):
                    check_interrupt()
            finally:
                state.event_loop = None
        return 42

    t, res = run_worker(fn, thread_cls=HostedThread)
    t.join(3)
    assert not t.is_alive()
    assert res.exc is None
    assert res.value == 42


def test_scope_rejects_another_thread(hosted_patches):
    thread = HostedThread()
    with pytest.raises(
        RuntimeError, match="current thread"
    ), thread.interruption_scope():
        pytest.fail("entered a scope for another thread")


def test_scope_rejects_existing_registration_without_clearing_it(hosted_patches):
    def fn(res):
        thread = threading.current_thread()
        with critical_section():
            thread.interrupt()
            with pytest.raises(
                RuntimeError, match="already registered"
            ), thread.interruption_scope():
                pytest.fail("replaced an existing registration")
            assert clear_interrupt()
        return 42

    t, res = run_worker(fn)
    t.join(3)
    assert not t.is_alive()
    assert res.exc is None
    assert res.value == 42


def test_scoped_async_command_preserves_host_loop(hosted_patches):
    import asyncio

    ready = queue.Queue()

    def fn(res):
        async def host():
            release_background = asyncio.Event()
            background = asyncio.create_task(release_background.wait())

            async def command():
                task = asyncio.current_task()
                with threading.current_thread().interruption_scope(
                    cooperative_only=True,
                    should_interrupt=lambda: asyncio.current_task() is task,
                ):
                    ready.put((asyncio.get_running_loop(), task))
                    await asyncio.Event().wait()

            task = asyncio.create_task(command())
            try:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert not background.done()
                release_background.set()
                assert await background
                with threading.current_thread().interruption_scope(
                    cooperative_only=True
                ):
                    await asyncio.sleep(0)
                    check_interrupt()
                return 42
            finally:
                background.cancel()

        # The scope must coexist with a host loop that already exists.
        return asyncio.run(host())

    t, res = run_worker(fn, thread_cls=HostedThread)
    loop, task = ready.get(timeout=2)
    t.interrupt()
    callback_errors = []

    def cancel_command():
        try:
            # This callback runs on the same thread, outside the command task.
            check_interrupt()
            assert is_interrupted()
        except BaseException as error:  # noqa: BLE001 - report to the test
            callback_errors.append(error)
        finally:
            task.cancel()

    loop.call_soon_threadsafe(cancel_command)
    t.join(3)
    assert not t.is_alive()
    assert not callback_errors
    assert res.exc is None
    assert res.value == 42


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
