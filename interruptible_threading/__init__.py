"""
Cooperative thread interruption for CPython.

Exposes an :class:`InterruptibleThread` whose :meth:`~InterruptibleThread.interrupt`
method raises an exception (``ThreadInterrupted`` by default) inside the target
thread. Pure-Python code is interrupted via ``PyThreadState_SetAsyncExc`` (an async
exception that fires at the next bytecode boundary). Because async exceptions cannot
break a thread that is parked in a C-level blocking call, ``install_patches()``
monkeypatches a curated set of stdlib blocking primitives (``time.sleep``,
``selectors.DefaultSelector``, ``select.select``, ``threading.Condition.wait``) so
they wake promptly and re-check a durable per-thread "interrupt pending" flag.

The flag is the single source of truth: ``interrupt()`` sets it first (under one
lock) and then issues a wakeup nudge; every blocking primitive checks the flag
before parking and again after waking. This closes the races inherent in choosing a
delivery path from transient state, and lets interrupts be *masked* during critical
sections (``critical_section()``) and *polled* in CPU-bound loops (``check_interrupt()``).

Requires monkey patching some stdlib functionality via
``InterruptibleThread.install_patches()``. CPython only; POSIX (Linux / Darwin).

Why not signals: ``signal.pthread_kill`` can unblock a syscall but cannot deliver an
exception to a worker thread -- CPython runs Python-level signal handlers only on the
main thread, and PEP 475's EINTR auto-retry loops call ``PyErr_CheckSignals()``
(a no-op off the main thread) without consulting ``tstate->async_exc``, so the
syscall is transparently retried. The self-pipe + cooperative-primitive approach is
the only way to get prompt, exception-bearing interruption of worker threads.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import select
import selectors
import socket as _socket
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from io import IOBase

__all__ = [
    "InterruptibleThread",
    "ThreadInterrupted",
    "is_interrupted",
    "clear_interrupt",
    "check_interrupt",
    "interruptible_checkpoint",
    "periodic_checkpoint",
    "critical_section",
    "interrupts_disabled",
    "interruptible_recv",
    "interruptible_send",
    "interruptible_accept",
    "set_poll_interval",
]


class ThreadInterrupted(BaseException):
    """Raised inside a thread when another thread calls ``.interrupt()`` on it.

    Subclasses ``BaseException`` (not ``Exception``) so that ordinary
    ``except Exception:`` handlers do not accidentally swallow it -- mirroring
    ``KeyboardInterrupt`` and ``asyncio.CancelledError``.
    """


class _ThreadInterruptedKeyboard(ThreadInterrupted, KeyboardInterrupt):
    """Legacy-compatible interrupt: caught by both ``ThreadInterrupted`` and
    ``KeyboardInterrupt`` handlers. Used when ``legacy_keyboardinterrupt=True``."""


# The exception class delivered by interrupt(); swapped by install_patches().
_INTERRUPT_EXC: type[BaseException] = ThreadInterrupted

_PTSSE = ctypes.pythonapi.PyThreadState_SetAsyncExc
_PTSSE.argtypes = [ctypes.c_ulong, ctypes.py_object]
_PTSSE.restype = ctypes.c_int
_MAIN_IDENT = threading.main_thread().ident

_ORIG_SLEEP = time.sleep
_ORIG_DEFAULT_SELECTOR = selectors.DefaultSelector
_ORIG_SELECT = select.select
_ORIG_THREAD = threading.Thread
_ORIG_COND_WAIT = threading.Condition.wait

_WAKEUP_TOKEN = "__interruptible_wakeup__"

# Max latency (seconds) for chunked-poll primitives (Condition.wait / Event / Queue).
_POLL_INTERVAL = 0.05


def set_poll_interval(seconds: float) -> None:
    """Tune the wakeup latency of the chunked-poll blocking primitives."""
    global _POLL_INTERVAL
    _POLL_INTERVAL = float(seconds)


def _clear_async_exc(tid: int | None = None) -> None:
    """Clear any async exception armed for ``tid`` (default: current thread).

    Called right before raising on a cooperative path so an exception armed by
    ``interrupt()`` does not fire a second time at the next bytecode boundary.
    """
    # An empty py_object() is a NULL pointer, which tells SetAsyncExc to *clear*
    # the pending async exception. Passing Python ``None`` would instead arm None
    # as the exception (raising SystemError when later delivered).
    _PTSSE(
        ctypes.c_ulong(tid if tid is not None else threading.get_ident()),
        ctypes.py_object(),
    )


def _tracing_active() -> bool:
    """Whether a trace/profile hook is installed (coverage, a debugger, a
    profiler) anywhere we can observe.

    ``PyThreadState_SetAsyncExc`` can permanently wedge a thread that is doing
    lock operations while such a hook is active (a CPython interaction), so async
    injection -- the only delivery path that uses it -- is skipped while tracing.
    Cooperative delivery via the durable ``pending`` flag is unaffected.
    """
    if sys.gettrace() is not None or sys.getprofile() is not None:
        return True
    # threading.gettrace/getprofile (3.10+) expose the global hook coverage
    # installs for worker threads; guarded for 3.9 where they don't exist.
    for name in ("gettrace", "getprofile"):
        getter = getattr(threading, name, None)
        if getter is not None and getter() is not None:
            return True
    return False


def _drain(fd: int) -> None:
    """Drain a non-blocking wakeup pipe until empty."""
    while True:
        try:
            if not os.read(fd, 65536):
                break
        except (BlockingIOError, OSError):
            break


class _State:
    registry_lock = threading.RLock()
    registry: dict[int, _State] = {}

    def __init__(self) -> None:
        cur_thread = threading.current_thread()
        if not isinstance(cur_thread, InterruptibleThread):
            raise TypeError(
                f"current thread should be of type {InterruptibleThread.__name__}"
            )
        self.thread: InterruptibleThread = cur_thread
        self.children: set[int] = set()
        # cancel_cond's lock is the single per-thread mutex guarding all the
        # mutable interrupt state below.
        self.cancel_cond = threading.Condition()
        # Durable source of truth: an interrupt has been requested, not yet delivered.
        self.pending = False
        self.interrupt_gen = 0
        self.mask_depth = 0
        # True only while an async exception has actually been armed for this
        # thread via PyThreadState_SetAsyncExc. Gates the ctypes "clear" call so
        # the cooperative paths never touch SetAsyncExc -- calling it while a
        # sys.settrace tracer (e.g. coverage) is active can deadlock the thread.
        self.async_armed = False
        self.cooperative_only = False
        self.should_interrupt: Callable[[], bool] | None = None
        # Hints used only to pick a wakeup nudge; never the source of truth.
        self.sleeping = False
        self.selecting = False
        # asyncio integration (set by run_interruptible).
        self.event_loop: Any = None
        self.root_task: Any = None
        # Lazily-allocated self-pipe; only needed on the selector/select path.
        self.rfd = -1
        self.wfd = -1

    def ensure_pipe(self) -> None:
        """Allocate the self-pipe on first use (idempotent)."""
        with self.cancel_cond:
            if self.rfd == -1:
                r, w = os.pipe()
                os.set_blocking(r, False)
                # The write end must also be non-blocking: ``_pipe_write`` runs
                # while holding ``cancel_cond``, and a blocking ``os.write`` to a
                # full pipe would deadlock. A full pipe already means a wakeup is
                # pending, so dropping the extra byte (EAGAIN) is correct.
                os.set_blocking(w, False)
                self.rfd, self.wfd = r, w

    @classmethod
    def register_current_thread(cls) -> _State:
        tid = threading.get_ident()
        with cls.registry_lock:
            st = cls.registry.get(tid)
            if st is None:
                st = cls()
                cls.registry[tid] = st
            return st

    @classmethod
    def get_state_by_ident(cls, tid: int | None = None) -> _State | None:
        return cls.registry.get(tid if tid is not None else threading.get_ident())

    @classmethod
    def unregister_current_thread(cls) -> None:
        tid = threading.get_ident()
        with cls.registry_lock:
            st = cls.registry.pop(tid, None)
            if not st:
                return
            for fd in (st.rfd, st.wfd):
                if fd == -1:
                    continue
                try:
                    os.close(fd)
                except OSError:
                    pass
            st.rfd = st.wfd = -1


def _disarm_async(st: _State) -> None:
    """Clear an armed async exception for the current thread, but only if one was
    actually armed. ``PyThreadState_SetAsyncExc`` must not be called speculatively:
    invoking it while a ``sys.settrace`` tracer (e.g. coverage) is active can wedge
    the thread, so the cooperative paths -- which never arm one -- must skip it.

    Must be called holding ``st.cancel_cond``.
    """
    if st.async_armed:
        st.async_armed = False
        _clear_async_exc()


def _take_pending(st: _State) -> bool:
    """If an unmasked interrupt is pending, consume it and clear any armed async
    exception, returning True (caller should raise). Otherwise return False.

    Must be called holding ``st.cancel_cond``.
    """
    if (
        st.pending
        and st.mask_depth == 0
        and (st.should_interrupt is None or st.should_interrupt())
    ):
        st.pending = False
        _disarm_async(st)
        return True
    return False


# --------------------------------------------------------------------------- #
# Public cooperative API (operates on the *current* thread)                    #
# --------------------------------------------------------------------------- #


def is_interrupted(thread: InterruptibleThread | None = None) -> bool:
    """Return whether an interrupt is pending for ``thread`` (default: current).

    Non-consuming; safe to call after the thread has exited (returns ``False``).
    """
    tid = thread.ident if thread is not None else threading.get_ident()
    st = _State.get_state_by_ident(tid)
    if st is None:
        return False
    with st.cancel_cond:
        return st.pending


def clear_interrupt() -> bool:
    """Consume any pending interrupt for the current thread without raising.

    Returns whether one was pending (Java's ``Thread.interrupted()`` semantics).

    Call this after *catching* ``ThreadInterrupted`` when you intend to keep
    running: the pending flag is durable (so an interrupt is never lost if the
    thread parks in a blocking call before async injection can fire), so without
    clearing it the next checkpoint or blocking primitive would re-raise.
    """
    st = _State.get_state_by_ident()
    if st is None:
        return False
    with st.cancel_cond:
        prev = st.pending
        st.pending = False
        if prev:
            _disarm_async(st)
        return prev


def check_interrupt() -> None:
    """Raise the interrupt exception if one is pending and not masked.

    Cheap; intended for CPU-bound loops and around opaque C calls so they become
    interruptible while still honoring ``critical_section()``.
    """
    st = _State.get_state_by_ident()
    if st is None:
        return
    with st.cancel_cond:
        if _take_pending(st):
            raise _INTERRUPT_EXC()


# Alias.
interruptible_checkpoint = check_interrupt


@contextlib.contextmanager
def periodic_checkpoint(every: int = 1000) -> Iterator[_Ticker]:
    """Yield a ticker whose ``.tick()`` runs :func:`check_interrupt` every ``every``
    calls, amortizing the per-iteration cost in a tight loop::

        with periodic_checkpoint(every=1000) as ck:
            for item in huge_iterable:
                ck.tick()
                crunch(item)
    """
    yield _Ticker(every)


class _Ticker:
    __slots__ = ("_every", "_n")

    def __init__(self, every: int) -> None:
        self._every = max(1, int(every))
        self._n = 0

    def tick(self) -> bool:
        """Count one iteration; every Nth call runs check_interrupt(). Returns
        whether the check ran this call."""
        self._n += 1
        if self._n >= self._every:
            self._n = 0
            check_interrupt()
            return True
        return False


@contextlib.contextmanager
def interrupts_disabled() -> Iterator[None]:
    """Defer interrupt delivery to the current thread for the duration of the block.

    An interrupt that arrives while masked is recorded and delivered exactly once
    when the outermost mask exits (re-entrant). Reliable for the flag-driven paths
    (sleep / select / Condition / checkpoints); an async exception already in flight
    from ``SetAsyncExc`` microseconds before entering cannot be recalled.
    """
    st = _State.get_state_by_ident()
    if st is None:
        yield
        return
    with st.cancel_cond:
        st.mask_depth += 1
    try:
        yield
    finally:
        raise_now = False
        with st.cancel_cond:
            st.mask_depth -= 1
            raise_now = _take_pending(st)
        if raise_now:
            raise _INTERRUPT_EXC()


# Alias.
critical_section = interrupts_disabled


# --------------------------------------------------------------------------- #
# Patched blocking primitives                                                  #
# --------------------------------------------------------------------------- #


def _coop_sleep(secs: float) -> None:
    tid = threading.get_ident()
    if tid == _MAIN_IDENT:
        return _ORIG_SLEEP(secs)
    st = _State.get_state_by_ident(tid)
    if st is None:
        return _ORIG_SLEEP(secs)
    deadline = time.monotonic() + secs
    with st.cancel_cond:
        while True:
            if _take_pending(st):
                raise _INTERRUPT_EXC()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            st.sleeping = True
            try:
                _ORIG_COND_WAIT(st.cancel_cond, remaining)
            finally:
                st.sleeping = False
            # Loop: an unmasked pending flag raises at the top; a masked one keeps
            # sleeping out the remaining time; spurious wakeups simply re-park.


def _patched_cond_wait(self: threading.Condition, timeout: float | None = None) -> bool:
    tid = threading.get_ident()
    if tid == _MAIN_IDENT:
        return _ORIG_COND_WAIT(self, timeout)
    st = _State.get_state_by_ident(tid)
    if st is None:
        return _ORIG_COND_WAIT(self, timeout)
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        with st.cancel_cond:
            if _take_pending(st):
                raise _INTERRUPT_EXC()
        if deadline is None:
            chunk = _POLL_INTERVAL
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            chunk = min(_POLL_INTERVAL, remaining)
        if _ORIG_COND_WAIT(self, chunk):
            return True
        # Timed-out chunk: loop to re-check the pending flag.


class _InterruptibleSelector(_ORIG_DEFAULT_SELECTOR):
    """DefaultSelector that also watches the current thread's self-pipe, so an
    interrupt can wake a parked ``select`` (this is what makes asyncio interruptible).
    """

    def __init__(self) -> None:
        super().__init__()
        st = self._ithr_st = _State.get_state_by_ident()
        if st is not None:
            st.ensure_pipe()
            try:
                super().register(st.rfd, selectors.EVENT_READ, data=_WAKEUP_TOKEN)
            except (KeyError, ValueError):
                pass

    def select(self, timeout: float | None = None):
        st = getattr(self, "_ithr_st", None)
        if st is None:
            return super().select(timeout)
        with st.cancel_cond:
            if _take_pending(st):
                raise _INTERRUPT_EXC()
            st.selecting = True
        try:
            events = super().select(timeout)
        finally:
            with st.cancel_cond:
                st.selecting = False
        out = []
        for key, mask in events:
            if key.data == _WAKEUP_TOKEN:
                _drain(key.fd)
                continue
            out.append((key, mask))
        with st.cancel_cond:
            if _take_pending(st):
                raise _INTERRUPT_EXC()
        return out


def _fileno(x: IOBase | int) -> int:
    try:
        return x.fileno()  # type: ignore[union-attr]
    except AttributeError:
        return int(x)  # type: ignore[arg-type]


def _patched_select(
    rlist: list[IOBase | int],
    wlist: list[IOBase | int],
    xlist: list[IOBase | int],
    timeout: float | None = None,
) -> tuple[list[IOBase | int], list[IOBase | int], list[IOBase | int]]:
    tid = threading.get_ident()
    if tid == _MAIN_IDENT:
        return _ORIG_SELECT(rlist, wlist, xlist, timeout)
    st = _State.get_state_by_ident(tid)
    if st is None:
        return _ORIG_SELECT(rlist, wlist, xlist, timeout)

    st.ensure_pipe()
    rfd = st.rfd

    def to_fd_list(lst: list[IOBase | int]) -> list[tuple[int, IOBase | int]]:
        return [(_fileno(o), o) for o in lst]

    rmap = to_fd_list(rlist)
    wmap = to_fd_list(wlist)
    xmap = to_fd_list(xlist)

    rfd_list = [fd for fd, _ in rmap]
    wfd_list = [fd for fd, _ in wmap]
    xfd_list = [fd for fd, _ in xmap]

    if rfd not in rfd_list:
        rfd_list.append(rfd)

    with st.cancel_cond:
        if _take_pending(st):
            raise _INTERRUPT_EXC()
        st.selecting = True
    try:
        rr, ww, xx = _ORIG_SELECT(rfd_list, wfd_list, xfd_list, timeout)
    finally:
        with st.cancel_cond:
            st.selecting = False

    if rfd in rr:
        _drain(rfd)
    with st.cancel_cond:
        if _take_pending(st):
            raise _INTERRUPT_EXC()

    def map_back(
        fd_list: list[int], fmap: list[tuple[int, IOBase | int]]
    ) -> list[IOBase | int]:
        fds = set(fd_list)
        return [obj for fd, obj in fmap if fd in fds]

    return (
        map_back(rr, rmap),
        map_back(ww, wmap),
        map_back(xx, xmap),
    )


# --------------------------------------------------------------------------- #
# Interruptible socket helpers (opt-in; reuse the self-pipe directly)          #
# --------------------------------------------------------------------------- #


def _interruptible_io(sock: _socket.socket, want_write: bool, op):
    tid = threading.get_ident()
    st = _State.get_state_by_ident(tid)
    if st is None or tid == _MAIN_IDENT:
        return op()
    st.ensure_pipe()
    prev_timeout = sock.gettimeout()
    sock.setblocking(False)
    try:
        while True:
            with st.cancel_cond:
                if _take_pending(st):
                    raise _INTERRUPT_EXC()
                st.selecting = True
            try:
                if want_write:
                    _rr, ww, _xx = _ORIG_SELECT([st.rfd], [sock], [])
                    ready = sock in ww
                else:
                    rr, _ww, _xx = _ORIG_SELECT([sock, st.rfd], [], [])
                    ready = sock in rr
            finally:
                with st.cancel_cond:
                    st.selecting = False
            _drain(st.rfd)
            with st.cancel_cond:
                if _take_pending(st):
                    raise _INTERRUPT_EXC()
            if ready:
                try:
                    return op()
                except (BlockingIOError, InterruptedError):
                    continue
    finally:
        sock.settimeout(prev_timeout)


def interruptible_recv(sock: _socket.socket, bufsize: int, flags: int = 0) -> bytes:
    """``sock.recv`` that raises the interrupt exception if interrupted while blocked."""
    return _interruptible_io(sock, False, lambda: sock.recv(bufsize, flags))


def interruptible_send(sock: _socket.socket, data: bytes, flags: int = 0) -> int:
    """``sock.send`` that raises the interrupt exception if interrupted while blocked."""
    return _interruptible_io(sock, True, lambda: sock.send(data, flags))


def interruptible_accept(sock: _socket.socket):
    """``sock.accept`` that raises the interrupt exception if interrupted while blocked."""
    return _interruptible_io(sock, False, sock.accept)


_ORIG_SOCK_RECV = _socket.socket.recv
_ORIG_SOCK_SEND = _socket.socket.send
_ORIG_SOCK_ACCEPT = _socket.socket.accept


def _sock_recv_patch(self, bufsize, flags=0):
    return interruptible_recv(self, bufsize, flags)


def _sock_send_patch(self, data, flags=0):
    return interruptible_send(self, data, flags)


def _sock_accept_patch(self):
    return interruptible_accept(self)


# --------------------------------------------------------------------------- #
# The thread class                                                             #
# --------------------------------------------------------------------------- #


class InterruptibleThread(_ORIG_THREAD):
    """Thread with a built-in cooperative :meth:`interrupt`."""

    _patches_installed = False
    _socket_patched = False
    _thread_patched = False
    _selector_patched = False

    @contextlib.contextmanager
    def interruption_scope(
        self,
        *,
        cooperative_only: bool = False,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> Iterator[None]:
        """Register one execution on a thread whose run() is owned by a host.

        Call from this thread with no existing registration. The ordinary run()
        registers for its entire lifetime, so this API is for subclasses that
        supply their own runner. Scopes cannot nest.

        cooperative_only disables async exception injection; CPU work must use
        checkpoints supplied by the caller. should_interrupt gates cooperative
        delivery and leaves the request pending when false. It runs on the target
        thread under the state lock and must not block or raise.
        """
        if self is not threading.current_thread():
            raise RuntimeError("interruption_scope must run on the current thread")
        if _State.get_state_by_ident() is not None:
            raise RuntimeError("current thread is already registered")
        st = _State.register_current_thread()
        st.cooperative_only = cooperative_only
        st.should_interrupt = should_interrupt
        try:
            yield
        finally:
            # Serialize teardown with a request that is waking a blocking call.
            with st.cancel_cond:
                clear_interrupt()
                _State.unregister_current_thread()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._parent_tid = threading.get_ident()

    def run(self) -> None:
        with _State.registry_lock:
            _State.register_current_thread()
            parent_st = _State.get_state_by_ident(self._parent_tid)
            if parent_st is not None:
                parent_st.children.add(threading.get_ident())
        try:
            super().run()
        finally:
            my_tid = threading.get_ident()
            with _State.registry_lock:
                parent_st = _State.get_state_by_ident(self._parent_tid)
                if parent_st is not None:
                    parent_st.children.discard(my_tid)
                st = _State.get_state_by_ident(my_tid)
            if st is not None:
                with st.cancel_cond:
                    _State.unregister_current_thread()

    @classmethod
    def run_interruptible(cls, coro):
        """Run a coroutine via ``asyncio.run`` such that ``interrupt()`` cancels the
        event loop's tasks cooperatively (clean ``finally``/``async with`` unwind)
        instead of injecting an exception into the selector. Returns the coroutine's
        result; re-raises the interrupt exception if interrupted."""
        import asyncio

        st = _State.get_state_by_ident()

        async def _runner():
            if st is not None:
                with st.cancel_cond:
                    st.event_loop = asyncio.get_running_loop()
                    st.root_task = asyncio.current_task()
            try:
                return await coro
            finally:
                if st is not None:
                    with st.cancel_cond:
                        st.event_loop = None
                        st.root_task = None

        try:
            return asyncio.run(_runner())
        except asyncio.CancelledError:
            if st is not None:
                with st.cancel_cond:
                    if _take_pending(st):
                        raise _INTERRUPT_EXC() from None
            raise

    def _inject_exc(self) -> None:
        tid = ctypes.c_ulong(self.ident or 0)
        rc = _PTSSE(tid, ctypes.py_object(_INTERRUPT_EXC))
        if rc == 0:
            raise ValueError("no such thread")
        elif rc > 1:
            _PTSSE(tid, ctypes.py_object())
            raise SystemError("SetAsyncExc affected multiple threads")

    @staticmethod
    def _pipe_write(st: _State) -> None:
        if st.wfd != -1:
            try:
                os.write(st.wfd, b"\x00")
            except OSError:
                pass

    def _nudge(self, st: _State) -> None:
        """Wake the target out of any current park without arming an async exception.
        Must be called holding ``st.cancel_cond``."""
        st.cancel_cond.notify_all()
        self._pipe_write(st)

    def _cancel_event_loop(self, st: _State) -> None:
        loop, task = st.event_loop, st.root_task

        def _cancel():
            if task is not None and not task.done():
                task.cancel()

        try:
            loop.call_soon_threadsafe(_cancel)
        except RuntimeError:
            # A host supplying checkpoints must retain its delivery policy even
            # when the loop has already closed.
            if not st.cooperative_only:
                self._inject_exc()

    def interrupt(self, recursive: bool = False) -> None:
        st = _State.get_state_by_ident(self.ident)
        if st is None:
            raise ValueError("no such thread")

        if recursive:
            with _State.registry_lock:
                children = list(st.children)
            for child_tid in children:
                with _State.registry_lock:
                    child_st = _State.get_state_by_ident(child_tid)
                    child_thread = child_st.thread if child_st is not None else None
                if (
                    child_thread is None
                    or child_thread.ident != child_tid
                    or not child_thread.is_alive()
                ):
                    continue
                try:
                    child_thread.interrupt(recursive=True)
                except ValueError:
                    continue

        with st.cancel_cond:
            if _State.get_state_by_ident(self.ident) is None:
                # Thread terminated while we waited on the lock; nothing to do.
                return
            if st.pending:
                # Already requested -- idempotent. Re-nudge in case a prior wakeup
                # was missed; never re-arm async injection.
                self._nudge(st)
                return
            st.pending = True
            st.interrupt_gen += 1

            if st.event_loop is not None:
                # Running an asyncio loop: cancel tasks cooperatively.
                self._cancel_event_loop(st)
            elif st.mask_depth > 0:
                # Masked: record + unblock so it loops to its mask check; the
                # pending flag is delivered when the mask exits.
                self._nudge(st)
            elif st.sleeping:
                st.cancel_cond.notify_all()
            elif st.selecting and st.wfd != -1:
                self._pipe_write(st)
            else:
                # Pure-Python execution, or a thread that has *decided* to block
                # but not yet committed (so the sleeping/selecting hints still
                # read False -- common under tracing/coverage). Async-inject to
                # break a CPU loop, and also nudge every cooperative channel in
                # case the thread is entering a primitive right now.
                #
                # Crucially, do NOT clear `pending`. It is the durable source of
                # truth: if async injection cannot fire (the thread parks in a
                # C-level wait before the next bytecode boundary), the primitive
                # still sees `pending` on its pre-block / post-wake check and
                # raises. The cooperative consumer (`_take_pending`) clears the
                # flag and disarms the async exception together, so delivery is
                # exactly-once via whichever path wins. (A thread that *catches*
                # an async-injected interrupt and continues must call
                # `clear_interrupt()` -- see its docstring.)
                self._nudge(st)
                # Async injection uses PyThreadState_SetAsyncExc, which can
                # deadlock the target under an active trace hook (coverage /
                # debuggers); skip it then. Any thread that reaches a checkpoint
                # or blocking primitive is still delivered via the durable flag;
                # only a checkpoint-less pure-Python loop is left uninterruptible
                # while tracing (a narrow, documented gap).
                if not st.cooperative_only and not _tracing_active():
                    self._inject_exc()
                    st.async_armed = True

    @classmethod
    def get_thread_cls_for_current_thread(cls, item: str) -> type[threading.Thread]:
        if item != "Thread":
            raise AttributeError("No attribute %s in module threading" % item)
        st = _State.get_state_by_ident()
        if st is None:
            return _ORIG_THREAD
        else:
            return cls

    @classmethod
    def install_patches(
        cls,
        interrupt_exc: type[BaseException] = ThreadInterrupted,
        legacy_keyboardinterrupt: bool = False,
        monkeypatch_socket: bool = False,
        monkeypatch_thread: bool = True,
        monkeypatch_selector: bool = True,
    ) -> None:
        """Install the stdlib monkeypatches that make blocking calls interruptible.

        ``interrupt_exc``: the exception class delivered by ``interrupt()``
        (default ``ThreadInterrupted``; pass ``KeyboardInterrupt`` for legacy code).
        ``legacy_keyboardinterrupt``: deliver an exception caught by *both*
        ``ThreadInterrupted`` and ``KeyboardInterrupt`` handlers.
        ``monkeypatch_socket``: also swap blocking ``socket.recv/send/accept`` for
        their interruptible variants process-wide (off by default; large blast radius).
        ``monkeypatch_thread``: let threads created by registered threads inherit
        interruption support. Disable when a host owns thread construction.
        ``monkeypatch_selector``: replace the default selector. Disable when a
        host owns an event loop and cancels individual tasks itself.
        """
        global _INTERRUPT_EXC
        if cls._patches_installed:
            raise ValueError("patches already installed")
        cls._patches_installed = True

        if legacy_keyboardinterrupt:
            _INTERRUPT_EXC = _ThreadInterruptedKeyboard
        else:
            _INTERRUPT_EXC = interrupt_exc

        time.sleep = _coop_sleep  # type: ignore
        if monkeypatch_selector:
            cls._selector_patched = True
            selectors.DefaultSelector = _InterruptibleSelector  # type: ignore
        select.select = _patched_select  # type: ignore
        threading.Condition.wait = _patched_cond_wait  # type: ignore

        if monkeypatch_socket:
            cls._socket_patched = True
            _socket.socket.recv = _sock_recv_patch  # type: ignore
            _socket.socket.send = _sock_send_patch  # type: ignore
            _socket.socket.accept = _sock_accept_patch  # type: ignore

        if monkeypatch_thread:
            cls._thread_patched = True
            del threading.Thread
            threading.__getattr__ = cls.get_thread_cls_for_current_thread  # type: ignore

        def __dir__():
            return list(threading.__dict__.keys()) + [_ORIG_THREAD.__name__]

        if monkeypatch_thread:
            threading.__dir__ = __dir__  # type: ignore

    @classmethod
    def uninstall_patches(cls) -> None:
        global _INTERRUPT_EXC
        if not cls._patches_installed:
            raise ValueError("patches not installed")
        cls._patches_installed = False
        time.sleep = _ORIG_SLEEP
        select.select = _ORIG_SELECT  # type: ignore
        if cls._selector_patched:
            selectors.DefaultSelector = _ORIG_DEFAULT_SELECTOR  # type: ignore
            cls._selector_patched = False
        threading.Condition.wait = _ORIG_COND_WAIT  # type: ignore
        if cls._socket_patched:
            _socket.socket.recv = _ORIG_SOCK_RECV  # type: ignore
            _socket.socket.send = _ORIG_SOCK_SEND  # type: ignore
            _socket.socket.accept = _ORIG_SOCK_ACCEPT  # type: ignore
            cls._socket_patched = False
        if cls._thread_patched:
            threading.Thread = _ORIG_THREAD  # type: ignore
            del threading.__getattr__
            del threading.__dir__
            cls._thread_patched = False
        _INTERRUPT_EXC = ThreadInterrupted


from . import _version  # noqa: E402

__version__ = _version.get_versions()["version"]
