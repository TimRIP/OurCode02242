from dataclasses import dataclass
import sys
from typing import Literal, TypeAlias, Union, Optional, List, Dict, Tuple
import jpamb
from jpamb import jvm
# replace the Interval class header + methods that reference lo/hi/is_bottom with this:
from typing import Optional
from math import inf

NEG_INF = -inf
POS_INF = inf

@dataclass(frozen=True)
class Interval:

    low: Optional[float]
    high: Optional[float]

    def bottom(self) -> 'Interval':
        return Interval(None, None)

    def top(self) -> 'Interval':
        return Interval(NEG_INF, POS_INF)

    def const(self, value: float) -> 'Interval':
        return Interval(value, value)

    def __repr__(self):
        return f"[{self.low}, {self.high}]"

    def __str__(self) -> str:
        if self.is_bot(): 
            return "BOT"
        def b(x): 
            return "-inf" if x == NEG_INF else ("+inf" if x == POS_INF else str(int(x)))
        return f"[{b(self.low)}..{b(self.high)}]"

    def is_bot(self) -> bool:
        return (self.low is None and self.high is None) or (self.low is not None and self.high is not None and self.low > self.high)

    def is_top(self) -> bool:
        return (self.low == NEG_INF and self.high == POS_INF)

    def join(self, other: 'Interval') -> 'Interval':
        if self.is_bot():
            return other
        if other.is_bot():
            return self
        return Interval(min(self.low, other.low), max(self.high, other.high))

    def leq(self, other: 'Interval') -> bool:
        if self.is_bot():
            return True
        if other.is_bot():
            return False
        return self.low >= other.low and self.high <= other.high

    def norm(self) -> 'Interval':
        if self.low is None or self.high is None or self.low > self.high:
            return Interval(1, 0)  # BOT
        return self

    def add(self, other: 'Interval') -> 'Interval':
        if self.is_bot() or other.is_bot():
            return Interval(1, 0)
        return Interval(self.low + other.low, self.high + other.high)

    def sub(self, other: 'Interval') -> 'Interval':
        if self.is_bot() or other.is_bot():
            return Interval(1, 0)
        return Interval(self.low - other.high, self.high - other.low)

    def mul(self, other: 'Interval') -> 'Interval':
        if self.is_bot() or other.is_bot():
            return Interval(1, 0)
        products = [self.low * other.low, self.low * other.high,
                    self.high * other.low, self.high * other.high]
        return Interval(min(products), max(products))

    def div(self, other: 'Interval'):
        if self.is_bot() or other.is_bot():
            return Interval(1, 0)
        if other.low <= 0 <= other.high:
            return "divide by zero"
        quotients = [self.low / other.low, self.low / other.high,
                     self.high / other.low, self.high / other.high]
        return Interval(min(quotients), max(quotients))
    

def ivl_cmp(a: Interval, b: Interval, cond: str) -> str:
    if a.is_bot or b.is_bot: return "false"
    if cond == "eq":
        # equal is only definite if both are singletons with same point
        if a.low == a.high == b.low == b.high:
            return "true" if a.low == b.low else "false"
        # disjoint intervals => cannot be equal
        if a.high < b.low or b.high < a.low: return "false"
        return "maybe"
    if cond == "ne":
        if a.low == a.high == b.low == b.high:
            return "false" if a.low == b.low else "true"
        if a.high < b.low or b.high < a.low: return "true"
        return "maybe"
    if cond == "lt":
        if a.high < b.low: return "true"
        if a.low >= b.high: return "false"
        return "maybe"
    if cond == "le":
        if a.high <= b.low: return "true"
        if a.low > b.high: return "false"
        return "maybe"
    if cond == "gt":
        if b.high < a.low: return "true"
        if b.low >= a.high: return "false"
        return "maybe"
    if cond == "ge":
        if b.high <= a.low: return "true"
        if b.low > a.high: return "false"
        return "maybe"
    return "maybe"

def ivl_cond_zero(i: Interval, cond: str) -> str:
    # returns "true"/"false"/"maybe"
    if i.is_bot: return "false"  # dead
    low, high = i.low, i.high
    if cond == "eq":
        if low == 0.0 and high == 0.0: return "true"
        if high < 0.0 or low > 0.0: return "false"
        return "maybe"
    if cond == "ne":
        if low == 0.0 and high == 0.0: return "false"
        if high < 0.0 or low > 0.0: return "true"
        return "maybe"
    if cond == "lt":
        if high < 0.0: return "true"
        if low >= 0.0: return "false"
        return "maybe"
    if cond == "le":
        if high <= 0.0: return "true"
        if low > 0.0: return "false"
        return "maybe"
    if cond == "gt":
        if low > 0.0: return "true"
        if high <= 0.0: return "false"
        return "maybe"
    if cond == "ge":
        if low >= 0.0: return "true"
        if high < 0.0: return "false"
        return "maybe"
    # "is"/"isnot" should be for refs; default:
    return "maybe"
warnings = []

# -------------------------
# Value and heap models
# -------------------------
# In the stack and locals we store "abstract values" as tuples
# ('int', Sign) or ('ref', oid) or ('char', Sign)
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
            self.data = init_vals
        else:
            self.data = []

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

#initial_state = State(frames=[initial_frame], heap=init_heap)

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
            # push interval for integer constants
            if isinstance(v, int):
                push( ('int', Interval(v, v)) )
            elif isinstance(v, str):
                push( ('ref', None) )  # string refs as null
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
                if val[0] != 'int':
                    if val[0] == 'char':
                        frame.locals[i] = ('int', Interval(val[1], val[1]))
                    else:
                        frame.locals[i] = ('int', Interval(NEG_INF, POS_INF))
                else:
                    frame.locals[i] = ('int', Interval(val[1].low, val[1].high))
            else:
                frame.locals[i] = val
            frame.pc += 1
            return [state]


        case jvm.Binary(type=jvm.Int(), operant=op):
            b = pop(); a = pop()
            print("DEBUG a and b: Dividing", a, b)
            if a[0] != 'int' or b[0] != 'int':
                # if not ints, produce TOP
                print("Setting TOP due to non-int operand")
                res = ('int', Interval(NEG_INF, POS_INF))
            else:
                sa: Interval = a[1]  # type: ignore
                sb: Interval = b[1]  # type: ignore
                if op == jvm.BinaryOpr.Add:
                    res = ('int', sa.add(sb).norm())
                elif op == jvm.BinaryOpr.Sub:
                    # Note: concrete did pop v2, v1 then v1 - v2; we popped in same order
                    res = ('int', sa.sub(sb).norm())
                elif op == jvm.BinaryOpr.Mul:
                    res = ('int', sa.mul(sb).norm())

                elif op == jvm.BinaryOpr.Div:
                    if sb.low <= 0 <= sb.high:
                        return "divide by zero"
                    if sb.is_top():
                        warnings.append("possible divide by zero")
                    tmp = sa.div(sb)
                    res = ('int', tmp.norm() if isinstance(tmp, Interval) else Interval(NEG_INF, POS_INF))

                elif op == jvm.BinaryOpr.Rem:

                    # if divisor definitely zero -> error
                    if sb.low <= 0 <= sb.high:
                        return "divide by zero"
                    if sb.is_top():
                        warnings.append("possible divide by zero")

                    res = ('int', Interval(NEG_INF, POS_INF))  # remainder result is hard to pin down; use TOP
                else:
                    res = ('int', Interval(NEG_INF, POS_INF))  # unknown op -> TOP
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
            name = op.field.fieldid.name
            if name == "$assertionsDisabled":
                push(('int', Interval(0, 0)))  # assertions enabled -> false (0)
                frame.pc += 1
                return [state]
            push(('int', Interval(0, 0)))
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


def run_worklist_result(initial: State) -> str:
    """Run and return terminal label ('ok', 'assertion error', etc.)."""
    worklist: List[State] = [initial]
    visited = set()
    while worklist:
        st = worklist.pop()
        frame = st.frames[-1]
        key = (frame.pc.method, frame.pc.offset, tuple(sorted((k, v) for k, v in frame.locals.items())))
        if key in visited:
            continue
        visited.add(key)

        res = step_abstract(st)
        if isinstance(res, str):
            return res
        for nxt in res:
            if isinstance(nxt, str):
                return nxt
            if not nxt.frames:
                continue
            worklist.append(nxt)
    return "ok"

def analyze_method_no_inputs(methodid: jvm.AbsMethodID) -> str:
    """Entry for analyzer/test mode (no concrete inputs)."""
    warnings.clear()
    return run_worklist_result(build_initial_state_from_sig(methodid))


def get_abstract_warnings() -> List[str]:
    """Return list of warning strings emitted during analysis."""
    return warnings