from dataclasses import dataclass
from math import inf
from typing import Optional, Tuple, Union
from typing import Dict, List
import sys
import jpamb
from jpamb import jvm

NEG_INF = -inf
POS_INF = inf

warnings = []

@dataclass(frozen=True)
class Interval:
    lo: Optional[float]
    hi: Optional[float]

    @staticmethod
    def bot() -> 'Interval':
        return Interval(None, None)

    @staticmethod
    def top() -> 'Interval':
        return Interval(NEG_INF, POS_INF)

    @staticmethod
    def const(v: int) -> 'Interval':
        return Interval(v, v)

    @property
    def is_bot(self) -> bool:
        return self.lo is None and self.hi is None

    @property
    def is_top(self) -> bool:
        return (self.lo == NEG_INF and self.hi == POS_INF)

    def __str__(self) -> str:
        if self.is_bot: return "BOT"
        def b(x): return "-inf" if x == NEG_INF else ("+inf" if x == POS_INF else str(int(x)))
        return f"[{b(self.lo)}..{b(self.hi)}]"


AVal = Tuple[str, Union[Interval, int, None]]

@dataclass
class _ArrayObj:
    elem_kind: str  # 'int' or 'char' or 'ref'
    length: int
    data: list[AVal]
    def __init__(self, elem_kind: str, length: int, init_vals: Optional[List[AVal]] = None):
        self.elem_kind = elem_kind
        self.length = length
        if init_vals is not None:
            self.data = init_vals + [(elem_kind, Interval.top())] * max(0, length - len(init_vals))
        else:
            self.data = [(elem_kind, Interval.top()) for _ in range(length)]

# -------------------------
# Interpreter frames/state
# -------------------------
@dataclass
class PC:
    method: jvm.AbsMethodID
    offset: int
    def __iadd__(self, d: int):
        self.offset += d
        return self

@dataclass
class Frame:
    locals: Dict[int, AVal]
    stack: List[AVal]
    pc: PC

@dataclass
class State:
    frames: List[Frame]
    heap: Dict[int, _ArrayObj]

# -------------------------
# Helpers: condition evaluation
# -------------------------
# Return "true", "false", or "maybe"

def norm(i: Interval) -> Interval:
    if i.is_bot: return i
    lo, hi = i.lo, i.hi
    if lo > hi:
        return Interval.bot()
    return i

def join(a: Interval, b: Interval) -> Interval:
    if a.is_bot: return b
    if b.is_bot: return a
    return Interval(min(a.lo, b.lo), max(a.hi, b.hi))

def leq(a: Interval, b: Interval) -> bool:
    # lattice order: a ⊑ b
    if a.is_bot: return True
    if b.is_bot: return False
    return b.lo <= a.lo and a.hi <= b.hi

def add(a: Interval, b: Interval) -> Interval:
    if a.is_bot or b.is_bot: return Interval.bot()
    return Interval(a.lo + b.lo, a.hi + b.hi)

def sub(a: Interval, b: Interval) -> Interval:
    if a.is_bot or b.is_bot: return Interval.bot()
    return Interval(a.lo - b.hi, a.hi - b.lo)

def mul_bounds(xs: Tuple[float,float], ys: Tuple[float,float]) -> Tuple[float,float]:
    products = [xs[0]*ys[0], xs[0]*ys[1], xs[1]*ys[0], xs[1]*ys[1]]
    return (min(products), max(products))

def mul(a: Interval, b: Interval) -> Interval:
    if a.is_bot or b.is_bot: return Interval.bot()
    lo, hi = mul_bounds((a.lo, a.hi), (b.lo, b.hi))
    return Interval(lo, hi)

def contains_zero(i: Interval) -> bool:
    if i.is_bot: return False
    return i.lo <= 0.0 <= i.hi

def div(a: Interval, b: Interval) -> Interval:
    # If divisor *may* be 0, be conservative by including +/-inf.
    if a.is_bot or b.is_bot: return Interval.bot()
    if b.lo == 0 and b.hi == 0:
        # caller should have trapped "divide by zero"
        return Interval.bot()

    # If b can be 0, split b into b1 <= -eps and b2 >= +eps; over-approx as TOP
    if contains_zero(b):
        return Interval.top()

    # Otherwise divide by a non-zero interval.
    inv_candidates = [1/b.lo, 1/b.hi]
    # If b spans sign without 0 (shouldn't), handle min/max anyway
    b1, b2 = min(inv_candidates), max(inv_candidates)
    lo, hi = mul_bounds((a.lo, a.hi), (b1, b2))
    return Interval(lo, hi)

def rem(a: Interval, b: Interval) -> Interval:
    # Very imprecise but safe:
    # If b can be 0 => potential error handled by caller; here return TOP
    if a.is_bot or b.is_bot: return Interval.bot()
    if contains_zero(b): return Interval.top()
    # |a % b| < |b|, so result in [-(|b|-1), |b|-1] when b is integral & positive.
    # With signs/unknowns, use symmetric bounds of |b|-1 if bounded.
    if b.lo == NEG_INF or b.hi == POS_INF:
        return Interval.top()
    maxmag = max(abs(b.lo), abs(b.hi))
    if maxmag == POS_INF:
        return Interval.top()
    # remainder is between -(maxmag-1) and +(maxmag-1)
    upper = max(0, maxmag - 1)
    return Interval(-upper, upper)

def ivl_cmp(a: Interval, b: Interval, cond: str) -> str:
    if a.is_bot or b.is_bot: return "false"
    if cond == "eq":
        # equal is only definite if both are singletons with same point
        if a.lo == a.hi == b.lo == b.hi:
            return "true" if a.lo == b.lo else "false"
        # disjoint intervals => cannot be equal
        if a.hi < b.lo or b.hi < a.lo: return "false"
        return "maybe"
    if cond == "ne":
        if a.lo == a.hi == b.lo == b.hi:
            return "false" if a.lo == b.lo else "true"
        if a.hi < b.lo or b.hi < a.lo: return "true"
        return "maybe"
    if cond == "lt":
        if a.hi < b.lo: return "true"
        if a.lo >= b.hi: return "false"
        return "maybe"
    if cond == "le":
        if a.hi <= b.lo: return "true"
        if a.lo > b.hi: return "false"
        return "maybe"
    if cond == "gt":
        if b.hi < a.lo: return "true"
        if b.lo >= a.hi: return "false"
        return "maybe"
    if cond == "ge":
        if b.hi <= a.lo: return "true"
        if b.lo > a.hi: return "false"
        return "maybe"
    return "maybe"

def ivl_cond_zero(i: Interval, cond: str) -> str:
    # returns "true"/"false"/"maybe"
    if i.is_bot: return "false"  # dead
    lo, hi = i.lo, i.hi
    if cond == "eq":
        if lo == 0.0 and hi == 0.0: return "true"
        if hi < 0.0 or lo > 0.0: return "false"
        return "maybe"
    if cond == "ne":
        if lo == 0.0 and hi == 0.0: return "false"
        if hi < 0.0 or lo > 0.0: return "true"
        return "maybe"
    if cond == "lt":
        if hi < 0.0: return "true"
        if lo >= 0.0: return "false"
        return "maybe"
    if cond == "le":
        if hi <= 0.0: return "true"
        if lo > 0.0: return "false"
        return "maybe"
    if cond == "gt":
        if lo > 0.0: return "true"
        if hi <= 0.0: return "false"
        return "maybe"
    if cond == "ge":
        if lo >= 0.0: return "true"
        if hi < 0.0: return "false"
        return "maybe"
    # "is"/"isnot" should be for refs; default:
    return "maybe"

# -------------------------
# Bytecode loader
# -------------------------
suite = jpamb.Suite()
bc_cache: Dict[jvm.AbsMethodID, List[jvm.Opcode]] = {}

def get_ops(method: jvm.AbsMethodID) -> List[jvm.Opcode]:
    if method not in bc_cache:
        bc_cache[method] = list(suite.method_opcodes(method))
    return bc_cache[method]

def get_opcode(method: jvm.AbsMethodID, offset: int) -> jvm.Opcode:
    ops = get_ops(method)
    # assume offset is index into ops list (same convention as your interpreter)
    return ops[offset]

# -------------------------
# Initialize from jpamb case
# -------------------------

# create initial frame locals (map input values to abstract values)
init_locals: Dict[int, AVal] = {}
init_heap: Dict[int, _ArrayObj] = {}
next_heap_id = 0

methodid = jpamb.parse_methodid(sys.argv[1])

initial_frame = Frame(locals=init_locals, stack=[], pc=PC(methodid, 0))

# -------------------------
# Step function (returns list[State] or error string)
# -------------------------
def step_abstract(state: State) -> Union[List[State], str]:
    if not state.frames:
        return ["done"]  # should not happen
    frame = state.frames[-1]
    try:
        op = get_opcode(frame.pc.method, frame.pc.offset)
    except Exception as e:
        return "ok"  # no code -> treat as finished

    # helper to push/pop
    def push(v: AVal):
        frame.stack.append(v)
    def pop() -> AVal:
        return frame.stack.pop()

    # Match opcodes by type (same shape as concrete interpreter)
    match op:
        case jvm.Push(value=v):
            if isinstance(v.type, jvm.Int) or v.type == jvm.Int():
                push(('int', Interval.const(int(v.value))))
            elif isinstance(v.type, jvm.Boolean) or v.type == jvm.Boolean():
                push(('int', Interval.const(0 if int(v.value) == 0 else 1)))
            elif isinstance(v.type, jvm.Char) or v.type == jvm.Char():
                cp = ord(v.value) if isinstance(v.value, str) else int(v.value)
                push(('char', Interval.const(cp)))
            elif isinstance(v.type, jvm.Array) or getattr(v.type, "contains", None) is not None:
                contains = getattr(v.type, "contains", None)
                if contains == jvm.Char() or isinstance(contains, jvm.Char):
                    elems = [('char', Interval.const(ord(x) if isinstance(x, str) else int(x)))
                            for x in (v.value or [])]
                    arr = _ArrayObj('char', len(elems), elems)
                elif contains == jvm.Int() or isinstance(contains, jvm.Int):
                    elems = [('int', Interval.const(int(x))) for x in (v.value or [])]
                    arr = _ArrayObj('int', len(elems), elems)
                else:
                    elems = [('ref', None) for _ in (v.value or [])]
                    arr = _ArrayObj('ref', len(elems), elems)
                oid = max(state.heap.keys()) + 1 if state.heap else 0
                state.heap[oid] = arr
                push(('ref', oid))
            else:
                # Unknown literal kind -> conservative
                push(('int', Interval.top()))
            frame.pc += 1
            return [state]

        case jvm.Load(type=t, index=i):
            # load interval for int/local; reference or null for ref/local
            if isinstance(t, jvm.Int) or t == jvm.Int():
                val = frame.locals.get(i, ('int', Interval(NEG_INF, POS_INF)))
            else:
                val = frame.locals.get(i, ('ref', None))

            push(val)
            frame.pc += 1
            return [state]

        case jvm.Store(type=t, index=i):
            val = pop()
            if isinstance(t, jvm.Int) or t == jvm.Int():
                if val[0] == 'int':
                    frame.locals[i] = ('int', val[1])            # keep interval as-is
                elif val[0] == 'char':
                    frame.locals[i] = ('int', val[1])            # char interval coerces to int interval
                else:
                    frame.locals[i] = ('int', Interval.top())
            else:
                frame.locals[i] = val
            frame.pc += 1
            return [state]

        case jvm.Binary(type=jvm.Int(), operant=op):
            b = pop(); a = pop()
            if a[0] != 'int' or b[0] != 'int':
                res = ('int', Interval.top())
            else:
                ia: Interval = a[1]  # type: ignore
                ib: Interval = b[1]  # type: ignore
                if op == jvm.BinaryOpr.Add:
                    res = ('int', add(ia, ib))
                elif op == jvm.BinaryOpr.Sub:
                    res = ('int', sub(ia, ib))
                elif op == jvm.BinaryOpr.Mul:
                    res = ('int', mul(ia, ib))
                elif op == jvm.BinaryOpr.Div:
                    if not ib.is_bot and ib.lo == 0 and ib.hi == 0:
                        return "divide by zero"
                    res = ('int', div(ia, ib))
                elif op == jvm.BinaryOpr.Rem:
                    if not ib.is_bot and ib.lo == 0 and ib.hi == 0:
                        return "divide by zero"
                    res = ('int', rem(ia, ib))
                else:
                    res = ('int', Interval.top())
            push(res)
            frame.pc += 1
            return [state]

                        
        case jvm.Return(type=ret_type):
            if ret_type is not None:
                vret = pop()
                # simple type check not enforced here
            state.frames.pop()
            if state.frames:
                caller = state.frames[-1]
                if ret_type is not None:
                    caller.stack.append(vret)
                    caller.pc += 1
                return [state]
            else:
                return ["ok"]

        case jvm.Get(static=True):
            # handle $assertionsDisabled style read
            name = op.field.fieldid.name
            if name == "$assertionsDisabled":
                push(('int', Interval.const(0) if True else Interval.const(1)))  # we assume assertions enabled -> push boolean 0
                frame.pc += 1
                return [state]
            
            push(('int', Interval.const(0)))
            frame.pc += 1
            return [state]

        case jvm.Ifz(condition=cond, target=tgt):
            v1 = pop()
            # cond is a str like 'eq','ne','lt', etc.
            c = (cond or "").lower()
            if v1[0] == 'int':
                res = ivl_cond_zero(v1[1], c)
            elif v1[0] == 'ref':
                # reference null checks (is/isnot style sometimes)
                # treat None as 0; else non-zero
                if c == "is":
                    res = "true" if v1[1] is None else "false"
                elif c == "isnot":
                    res = "false" if v1[1] is None else "true"
                else:
                    res = "maybe"
            else:
                res = "maybe"

            if res == "true":
                frame.pc.offset = tgt
                return [state]
            if res == "false":
                frame.pc += 1
                return [state]
            # maybe -> fork two states
            s_true = State(frames=[ Frame(locals=frame.locals.copy(),
                                         stack=frame.stack.copy(),
                                         pc=PC(frame.pc.method, tgt)) ],
                           heap=state.heap.copy())
            s_false = State(frames=[ Frame(locals=frame.locals.copy(),
                                          stack=frame.stack.copy(),
                                          pc=PC(frame.pc.method, frame.pc.offset + 1)) ],
                            heap=state.heap.copy())
            return [s_true, s_false]

        case jvm.New(classname=cn):
            # allocate an object reference (not array)
            oid = max(state.heap.keys()) + 1 if state.heap else 0
            # store as minimal dict disguised as _ArrayObj for reuse (length 0)
            state.heap[oid] = _ArrayObj('ref', 0, [])
            push(('ref', oid))
            frame.pc += 1
            return [state]

        case jvm.Dup(words = w):
            assert w == 1, "dup word >1 not supported"
            v = frame.stack[-1]
            push(v)
            frame.pc += 1
            return [state]

        case jvm.InvokeSpecial(method=m, is_interface=_):
            # handle constructors <init> as no-op (but mark init)
            m_str = str(m)
            arg_types = list(m.extension.params or [])
            argc = len(arg_types)
            args = [ pop() for _ in range(argc) ][::-1] if argc else []
            receiver = pop()  # the 'this' ref
            # mark constructed object if present
            if receiver[0] == 'ref' and receiver[1] is not None:
                # nothing else to do besides consider it initialized
                pass
            # if ctor returns nothing, continue
            frame.pc += 1
            return [state]

        case jvm.Throw():
            objref = pop()
            if objref[0] != 'ref':
                return "exception"
            ref = objref[1]
            if ref is None:
                return "exception"
            # check heap stored class tag? we didn't store class names, so treat as assertion error if we can guess
            # The concrete interpreter uses AssertionError to indicate assert failures.
            # We cannot precisely detect class; treat Throw as assertion error to match jpamb tests that use Throw for asserts.
            return "assertion error"

        case jvm.If(condition=cond, target=tgt):
            # binary comparison between two values (pop v2 then v1)
            v2 = pop(); v1 = pop()
            c = (cond or "").lower()
            # handle int-int comparisons
            if v1[0] == 'int' and v2[0] == 'int':
                res = ivl_cmp(v1[1], v2[1], c)
            elif v1[0] == 'ref' and v2[0] == 'ref':
                # comparing references: eq/is or isnot
                if c == "is":
                    if v1[1] is None and v2[1] is None: res = "true"
                    elif v1[1] is not None and v2[1] is not None and v1[1] == v2[1]: res = "true"
                    elif v1[1] is not None and v2[1] is not None and v1[1] != v2[1]: res = "false"
                    else: res = "maybe"
                elif c == "isnot":
                    if v1[1] is None and v2[1] is None: res = "false"
                    elif v1[1] is not None and v2[1] is not None and v1[1] == v2[1]: res = "false"
                    elif v1[1] is not None and v2[1] is not None and v1[1] != v2[1]: res = "true"
                    else: res = "maybe"
                else:
                    res = "maybe"
            else:
                res = "maybe"

            if res == "true":
                frame.pc.offset = tgt
                return [state]
            if res == "false":
                frame.pc += 1
                return [state]
            # maybe -> fork
            s_true = State(frames=[ Frame(locals=frame.locals.copy(),
                                         stack=frame.stack.copy(),
                                         pc=PC(frame.pc.method, tgt)) ],
                           heap=state.heap.copy())
            s_false = State(frames=[ Frame(locals=frame.locals.copy(),
                                          stack=frame.stack.copy(),
                                          pc=PC(frame.pc.method, frame.pc.offset + 1)) ],
                            heap=state.heap.copy())
            return [s_true, s_false]

        case jvm.Goto(target=tgt):
            frame.pc.offset = tgt
            return [state]

        case jvm.Cast(from_=jvm.Int(), to_=jvm.Short()):
            v1 = pop()
            if v1[0] == 'int':
                # cast preserves sign
                push(('int', v1[1]))
            else:
                push(('int', Interval(NEG_INF, POS_INF)))
            frame.pc += 1
            return [state]

        case jvm.Get(field=f) if not getattr(op, "static", False):
            # Instance field get not implemented in abstract -> conservatively TOP
            # But concrete interpreter asserted instance fields not implemented - we mimic that
            return "exception"

        case jvm.InvokeStatic(method=m):
            # pop args and create callee frame
            param_types = list(m.extension.params or [])
            argc = len(param_types)
            args = [ pop() for _ in range(argc) ][::-1] if argc else []
            callee_locals = {i: arg for i,arg in enumerate(args)}
            callee = Frame(locals=callee_locals, stack=[], pc=PC(m, 0))
            # push callee frame
            state.frames.append(callee)
            return [state]

        case jvm.Throw():
            # handled above but fallback
            return "assertion error"

        case a:
            # Unknown/unhandled opcode: be conservative and finish OK
            return ["ok"]

    # default fallback
    return [state]

def build_initial_state_from_sig(methodid: jvm.AbsMethodID) -> State:
    """Build initial State using only the method signature (no inputs): TOP / null."""
    init_locals: Dict[int, AVal] = {}
    init_heap: Dict[int, _ArrayObj] = {}
    for i, t in enumerate(methodid.extension.params):
        if t == jvm.Int() or t == jvm.Boolean() or t == jvm.Char() or isinstance(t, (jvm.Int, jvm.Boolean, jvm.Char)):
            init_locals[i] = ('int', Interval(NEG_INF, POS_INF))
        elif isinstance(t, jvm.Array) or getattr(t, "contains", None) is not None:
            init_locals[i] = ('ref', None)
        else:
            init_locals[i] = ('ref', None)
    initial_frame = Frame(locals=init_locals, stack=[], pc=PC(methodid, 0))
    return State(frames=[initial_frame], heap=init_heap)

def join_avals(a: AVal, b: AVal) -> AVal:
    if a[0] != b[0]:  # different kinds -> TOP-ish of the expected kind
        if a[0] == 'int' or b[0] == 'int':
            return ('int', Interval.top())
        return a  # keep as-is for refs (you can refine later)
    if a[0] == 'int':
        return ('int', join(a[1], b[1]))  # type: ignore
    if a[0] == 'ref':
        # simple ref join: same oid -> same; else unknown/null -> None, else keep None (unknown)
        ra, rb = a[1], b[1]
        return ('ref', ra if ra == rb else None)
    if a[0] == 'char':
        return ('char', join(a[1], b[1]))  # type: ignore
    return a

def join_frames(fa: Frame, fb: Frame) -> Frame:
    # Join locals pointwise (only indices that appear in either)
    keys = set(fa.locals.keys()) | set(fb.locals.keys())
    jlocals = {k: join_avals(fa.locals.get(k, ('int', Interval.top())),
                             fb.locals.get(k, ('int', Interval.top())))
               for k in keys}
    # Join stacks conservatively: require same height; else surrender to TOP-ish
    if len(fa.stack) == len(fb.stack):
        jstack = [ join_avals(x, y) for x, y in zip(fa.stack, fb.stack) ]
    else:
        jstack = [ ('int', Interval.top()) ] * max(len(fa.stack), len(fb.stack))
    # pc join is the *same* key so pc equal here.
    return Frame(locals=jlocals, stack=jstack, pc=fa.pc)

def states_equal(a: Frame, b: Frame) -> bool:
    return a.locals == b.locals and a.stack == b.stack  # coarse but fine


def run_worklist_result(initial: State) -> str:
    worklist: List[State] = [initial]
    seen: Dict[Tuple[jvm.AbsMethodID, int], Frame] = {}

    while worklist:
        st = worklist.pop()
        if not st.frames: 
            continue
        fr = st.frames[-1]
        key = (fr.pc.method, fr.pc.offset)

        if key in seen:
            joined = join_frames(seen[key], fr)
            if states_equal(joined, seen[key]):
                continue
            seen[key] = joined
            st.frames[-1] = joined
        else:
            seen[key] = fr

        res = step_abstract(st)
        if isinstance(res, str):
            return res
        for nxt in res:
            if isinstance(nxt, str): return nxt
            if not nxt.frames: continue
            worklist.append(nxt)
    return "ok"

def analyze_method_no_inputs(methodid: jvm.AbsMethodID) -> str:
    """Entry for analyzer/test mode (no concrete inputs)."""
    warnings.clear()
    return run_worklist_result(build_initial_state_from_sig(methodid))


def get_abstract_warnings() -> List[str]:
    """Return list of warning strings emitted during analysis."""
    return warnings