from dataclasses import dataclass
from typing import Literal, TypeAlias, Union, Optional, List, Dict, Tuple
from unittest import case
import jpamb
import numbers
from jpamb import jvm

# -------------------------
# Sign lattice definitions
# -------------------------
Sign: TypeAlias = Literal["+", "-", "0", "NZ", "TOP", "BOT"]

def sign_of_int(x: int) -> Sign:
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"

def sign_join(a: Sign, b: Sign) -> Sign:
    if a == b: return a
    if a == "BOT": return b
    if b == "BOT": return a
    # unions:
    A, B = _to_set(a), _to_set(b)
    return _from_set(A | B)

def sign_add(a: Sign, b: Sign) -> Sign:
    if "BOT" in (a,b): return "BOT"
    if "TOP" in (a,b): return "TOP"
    if a == "0": return b
    if b == "0": return a
    # + + + = + ; - + - = -
    if a == b and a in {"+","-"}: return a
    # any mix with NZ or +/- conflict → could be -,0,+
    if "NZ" in (a,b) or (a in {"+","-"} and b in {"+","-"} and a != b):
        return "TOP"
    return "TOP"

def sign_sub(a: Sign, b: Sign) -> Sign:
    if "BOT" in (a,b): return "BOT"
    if "TOP" in (a,b): return "TOP"
    if b == "0": return a
    if a == "0":
        if b in {"+","-"}: return "-" if b == "+" else "+"
        if b == "NZ": return "TOP"
        return "0"
    if a in {"+","-"} and b in {"+","-"}:
        if a == b: return "TOP"    # could be 0
        return a                   # + - - = + ; - - + = -
    if b == "NZ": return "TOP"
    if a == "NZ": return "TOP"
    return "TOP"

def sign_mul(a: Sign, b: Sign) -> Sign:
    if "BOT" in (a,b): return "BOT"
    if "0" in (a,b): return "0"
    if "TOP" in (a,b): return "TOP"
    # Now a,b in {+,-,NZ}
    if a == "NZ" or b == "NZ":
        # NZ * (+/-/NZ) is NZ
        return "NZ"
    return "+" if a == b else "-"

def sign_div(a: Sign, b: Sign) -> Sign:
    # caller already checks b != 0
    if "BOT" in (a,b): return "BOT"
    if a == "0": return "0"
    if "TOP" in (a,b): return "TOP"
    # b in {+,-,NZ}
    if b == "NZ":
        if a in {"+","-","NZ"}: return "NZ"
        return "0"  # already handled a==0 above
    # b is + or -
    if a == "NZ": return "NZ"
    if a in {"+","-"}: return "+" if a == b else "-"
    return "TOP"

def sign_rem(sa: Sign, sb: Sign) -> Sign:
    # Caller already guards sb == "0"
    if sa == "0":
        return "0"
    # In Java/C#, remainder sign follows the dividend’s sign when divisor ≠ 0.
    # Keep it conservative for abstract signs:
    if sa in {"+", "-"}:
        return sa            # optional: return "NZ" if you use that lattice
    if sa == "NZ":
        return "NZ"
    return "TOP"
# -------------------------
# Value and heap models
# -------------------------
# In the stack and locals we store "abstract values" as tuples
# ('int', Sign) or ('ref', oid) or ('char', Sign)
AVal = Tuple[str, Union[Sign, int, None]]

@dataclass
class _ArrayObj:
    elem_kind: str  # 'int' or 'char' or 'ref'
    length: int
    data: list[AVal]
    def __init__(self, elem_kind: str, length: int, init_vals: Optional[List[AVal]] = None):
        self.elem_kind = elem_kind
        self.length = length
        if init_vals is not None:
            # if init_vals present, normalize to AVal tuples
            self.data = init_vals + [ (elem_kind, "TOP") ] * max(0, length - len(init_vals))
        else:
            self.data = [ (elem_kind, "TOP") for _ in range(length) ]

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


def mk_int(sign: str, const: int | None = None):
    # reduced product of Sign × Const
    if const is not None:
        # constant dictates sign
        sign = '0' if const == 0 else ('+' if const > 0 else '-')
        return ('int', sign, const)
    # optional tightening: if sign is '0', const must be 0
    if sign == '0':
        return ('int', '0', 0)
    return ('int', sign)

def _is_int(v) -> bool:
    return isinstance(v, tuple) and len(v) >= 2 and v[0] == 'int'

def _exact(v) -> int | None:
    """Return the concrete integer if known, else None.
    Works for ('int', sign) and ('int', sign, exact)."""
    if not (isinstance(v, tuple) and len(v) >= 2 and v[0] == 'int'):
        return None
    if len(v) >= 3 and isinstance(v[2], numbers.Integral):
        return int(v[2])
    return None

def _sign_of_exact(n: int) -> str:
    return '0' if n == 0 else ('+' if n > 0 else '-')

def _with_exact_if_known(sign: str, ex: int | None):
    return mk_int(_sign_of_exact(ex), ex) if ex is not None else ('int', sign)


def unary_sign_cond_eval(sign: Sign, cond: str) -> str:
    if cond == "eq":
        if sign == "0": return "true"
        if sign in {"+", "-", "NZ"}: return "false"
        return "maybe"
    if cond == "ne":
        if sign == "0": return "false"
        if sign in {"+", "-", "NZ"}: return "true"
        return "maybe"
    if cond == "lt":
        if sign == "-": return "true"
        if sign in {"0", "+", "NZ"}: return "false" if sign in {"+", "NZ"} else "false" if sign == "0" else "maybe"
        return "maybe"
    if cond == "le":
        if sign in {"-", "0"}: return "true"
        if sign in {"+", "NZ"}: return "false"
        return "maybe"
    if cond == "gt":
        if sign in {"+", "NZ"}: return "true" if sign == "+" else "maybe"
        if sign in {"0", "-"}: return "false"
        return "maybe"
    if cond == "ge":
        if sign in {"+", "0", "NZ"}: return "true" if sign in {"+", "0"} else "maybe"
        if sign == "-": return "false"
        return "maybe"
    return "maybe"


def compare_two_signs(a: Sign, b: Sign, cond: str) -> str:
    # cond: 'eq', 'ne', 'lt', 'le', 'gt', 'ge'
    if cond == "eq":
        if a == b and a != "TOP": return "true"
        if a != b and a != "TOP" and b != "TOP": return "false"
        return "maybe"
    if cond == "ne":
        if a == b and a != "TOP": return "false"
        if a != b and a != "TOP" and b != "TOP": return "true"
        return "maybe"
    if cond == "lt":
        # - < 0, - < +, 0 < + is true only for 0<+
        # definite checks
        if a == "-" and b in {"0","+","-","TOP"}:
            # a is negative; if b is "-" then uncertain, treat '-' < '-' as maybe (could be equal)
            if b == "-": return "maybe"
            return "true"
        if a == "0" and b == "+":
            return "true"
        if a in {"+","0"} and b == "-":
            return "false"
        if a == "+" and b == "0":
            return "false"
        return "maybe"
    if cond == "le":
        # a <= b true if lt or eq
        r_lt = compare_two_signs(a,b,"lt")
        r_eq = compare_two_signs(a,b,"eq")
        if r_lt == "true" or r_eq == "true": return "true"
        if r_lt == "false" and r_eq == "false": return "false"
        return "maybe"
    if cond == "gt":
        return compare_two_signs(b,a,"lt")
    if cond == "ge":
        return compare_two_signs(b,a,"le")
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
methodid, input_values = jpamb.getcase()

# create initial frame locals (map input values to abstract values)
init_locals: Dict[int, AVal] = {}
init_heap: Dict[int, _ArrayObj] = {}
next_heap_id = 0

for i, v in enumerate(input_values.values):
    # jvm types: jvm.Int(), jvm.Boolean(), jvm.Char(), jvm.Array(), etc.
    if isinstance(v.type, jvm.Int) or v.type == jvm.Int():
        init_locals[i] = mk_int(_sign_of_exact(v.value), v.value)
    elif isinstance(v.type, jvm.Boolean) or v.type == jvm.Boolean():
    # booleans are ints 0/1
        exact_bool = int(v.value)
        init_locals[i] = mk_int("0" if exact_bool == 0 else "+", exact_bool)

    elif isinstance(v.type, jvm.Char) or v.type == jvm.Char():
        # normalize char input to sign of its codepoint
        ch = v.value
        if isinstance(ch, str):
            init_locals[i] = ('char', sign_of_int(ord(ch)))
        else:
            init_locals[i] = ('char', sign_of_int(int(ch)))
    elif isinstance(v.type, jvm.Array) or getattr(v.type, "contains", None) is not None:
        # Construct an array object in heap and store its reference (oid) in locals
        # element kind detection:
        contains = getattr(v.type, "contains", None)
        if contains == jvm.Int() or isinstance(contains, jvm.Int):
            elems = [ ('int', sign_of_int(x)) for x in (v.value or []) ]
            arr = _ArrayObj('int', len(elems), elems)
        elif contains == jvm.Char() or isinstance(contains, jvm.Char):
            elems = [ ('char', sign_of_int(ord(x) if isinstance(x, str) else int(x))) for x in (v.value or []) ]
            arr = _ArrayObj('char', len(elems), elems)
        else:
            # references or unknown -> initialize TOP refs
            elems = [ ('ref', None) for _ in (v.value or []) ]
            arr = _ArrayObj('ref', len(elems), elems)
        oid = next_heap_id
        next_heap_id += 1
        init_heap[oid] = arr
        init_locals[i] = ('ref', oid)
    elif v.value is None:
        init_locals[i] = ('ref', None)
    else:
        # fallback
        init_locals[i] = ('int', 'TOP')

initial_frame = Frame(locals=init_locals, stack=[], pc=PC(methodid, 0))
initial_state = State(frames=[initial_frame], heap=init_heap)

# ---- exact-value helpers ----

def refine_local_in_state(st, idx: int, cond: str, branch_is_true: bool):
    """
    Narrow the abstract sign of local `idx` given a zero-comparison condition.
    Keep exact value when consistent; force exact 0 when narrowed to '0'.
    """
    if idx is None:
        return
    fr = st.frames[-1]
    val = fr.locals.get(idx, ('int', 'TOP'))
    if not (isinstance(val, tuple) and val and val[0] == 'int'):
        return
    old_sign = val[1]
    old_ex   = _exact(val)
    new_sign = narrow_by_zero_test(old_sign, cond, branch_is_true)

    # Keep / adjust exact value
    if new_sign == '0':
        fr.locals[idx] = mk_int('0', 0)
    else:
        # keep old exact only if it’s compatible with the new sign
        keep_ex = old_ex
        if keep_ex is not None:
            if new_sign == '+' and keep_ex <= 0: keep_ex = None
            elif new_sign == '-' and keep_ex >= 0: keep_ex = None
            elif new_sign == 'NZ' and keep_ex == 0: keep_ex = None
        fr.locals[idx] = mk_int(new_sign, keep_ex)

def _refine_stack_top_in_state(st, cond: str, branch_is_true: bool):
    """
    Narrow the abstract sign of the stack top (handles DUP patterns).
    """
    fr = st.frames[-1]
    if not fr.stack:
        return
    val = fr.stack[-1]
    if not (isinstance(val, tuple) and val and val[0] == 'int'):
        return
    old_sign = val[1]
    old_ex   = _exact(val)
    new_sign = narrow_by_zero_test(old_sign, cond, branch_is_true)

    if new_sign == '0':
        fr.stack[-1] = mk_int('0', 0)
    else:
        keep_ex = old_ex
        if keep_ex is not None:
            if new_sign == '+' and keep_ex <= 0: keep_ex = None
            elif new_sign == '-' and keep_ex >= 0: keep_ex = None
            elif new_sign == 'NZ' and keep_ex == 0: keep_ex = None
        fr.stack[-1] = mk_int(new_sign, keep_ex)


def find_tested_local_for_ifz(frame, max_steps: int = 8) -> int | None:
    """
    Look backward from PC to find the producer of the value tested by If/Ifz (x <op> 0).
    If it's an int local load, return its index; otherwise None.
    Robust: does not rely on jvm.Nop existing; handles DUP by class name.
    """
    off = frame.pc.offset
    for step in range(1, max_steps + 1):
        idx = off - step
        if idx < 0:
            break
        try:
            instr = get_opcode(frame.pc.method, idx)
        except Exception:
            break

        name = instr.__class__.__name__.lower()

        # Skip DUP variants if present (Dup, DupX1, DupX2, etc.)
        if name.startswith("dup"):
            continue

        # Direct int local load → found the tested local
        if isinstance(instr, jvm.Load) and (isinstance(instr.type, jvm.Int) or instr.type == jvm.Int()):
            return instr.index

        # If it’s pushing a constant int (iconst/ldc int), there’s no local to refine
        if isinstance(instr, jvm.Push) and isinstance(getattr(instr, "value", None), int):
            return None

        # If we hit arithmetic (e.g., isub), stop — value was computed, not a plain local
        if isinstance(instr, jvm.Binary):
            return None

        # Unknown producer → stop conservatively
        break

    return None


def _just_loaded_int_local(frame) -> int | None:
    off = frame.pc.offset
    try:
        prev = get_opcode(frame.pc.method, off - 1)
    except Exception:
        return None

    # Direct load just before Ifz
    if isinstance(prev, jvm.Load) and (isinstance(prev.type, jvm.Int) or prev.type == jvm.Int()):
        return prev.index

    # If the previous op was a binary int op, check one more back for a load
    if isinstance(prev, jvm.Binary) and (isinstance(prev.type, jvm.Int) or prev.type == jvm.Int()):
        try:
            prev2 = get_opcode(frame.pc.method, off - 2)
        except Exception:
            return None
        if isinstance(prev2, jvm.Load) and (isinstance(prev2.type, jvm.Int) or prev2.type == jvm.Int()):
            return prev2.index

    return None

def _to_set(s: Sign) -> set[str]:
    if s == "TOP":
        return {"+", "-", "0"}
    if s == "NZ":
        return {"+", "-"}
    if s in {"+", "-", "0"}:
        return {s}
    return set()  # BOT maps to empty

def _from_set(S: set[str]) -> Sign:
    if not S:
        return "BOT"
    if S == {"+", "-", "0"}:
        return "TOP"
    if S == {"+", "-"}:
        return "NZ"
    if len(S) == 1:
        return next(iter(S))  # '+', '-', or '0'
    # any other 2-element mixture with '0' becomes TOP
    return "TOP"

def narrow_by_zero_test(cur: Sign, op: str, branch_is_true: bool) -> Sign:
    S = _to_set(cur)

    true_sets = {
        "eq": {"0"},
        "ne": {"+", "-"},    # non-zero
        "lt": {"-"},
        "le": {"-", "0"},
        "gt": {"+"},
        "ge": {"+", "0"},
    }

    if op not in true_sets:
        return cur

    if branch_is_true:
        S2 = S & true_sets[op]
    else:
        S2 = S - true_sets[op]

    return _from_set(S2)
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
            # push abstract representation
            if isinstance(v.type, jvm.Int) or v.type == jvm.Int():
               push(mk_int(_sign_of_exact(v.value), v.value))
            elif isinstance(v.type, jvm.Boolean) or v.type == jvm.Boolean():
               exact_bool = int(v.value)
               push(mk_int("0" if exact_bool == 0 else "+", exact_bool))    
            elif isinstance(v.type, jvm.Char) or v.type == jvm.Char():
                if isinstance(v.value, str):
                    push(('char', sign_of_int(ord(v.value))))
                else:
                    push(('char', sign_of_int(int(v.value))))
            elif isinstance(v.type, jvm.Array) or getattr(v.type, "contains", None) is not None:
                # create concrete heap array for pushed literal arrays (rare)
                contains = getattr(v.type, "contains", None)
                if contains == jvm.Char() or isinstance(contains, jvm.Char):
                    elems = [ ('char', sign_of_int(ord(x) if isinstance(x, str) else int(x))) for x in (v.value or []) ]
                    arr = _ArrayObj('char', len(elems), elems)
                elif contains == jvm.Int() or isinstance(contains, jvm.Int):
                    elems = [ ('int', sign_of_int(int(x))) for x in (v.value or []) ]
                    arr = _ArrayObj('int', len(elems), elems)
                else:
                    elems = [ ('ref', None) for _ in (v.value or []) ]
                    arr = _ArrayObj('ref', len(elems), elems)
                oid = max(state.heap.keys()) + 1 if state.heap else 0
                state.heap[oid] = arr
                push(('ref', oid))
            else:
                push(('int', 'TOP'))
            frame.pc += 1
            return [state]

        case jvm.Load(type=t, index=i):
            # load from locals: just push the stored abstract value
            val = frame.locals.get(i, ('int', 'TOP'))
            push(val)
            frame.pc += 1
            return [state]

        case jvm.Store(type=t, index=i):
            val = pop()
            # if expecting int, ensure we store ('int', sign)
            if isinstance(t, jvm.Int) or t == jvm.Int():
                if val[0] != 'int':
                    # try coerce char->int
                    if val[0] == 'char':
                        frame.locals[i] = ('int', val[1])
                    else:
                        frame.locals[i] = ('int', 'TOP')
                else:
                    frame.locals[i] = val
            else:
                # reference or other: store as-is
                frame.locals[i] = val
            frame.pc += 1
            return [state]

        case jvm.Binary(type=jvm.Int(), operant=op):
            b = pop(); a = pop()
            
            if a[0] != 'int' or b[0] != 'int':
                res = ('int', 'TOP')
            else:
                sa: Sign = a[1]  # type: ignore
                sb: Sign = b[1]  # type: ignore
                ea, eb = _exact(a), _exact(b)   # exact ints if known, else None

                if op == jvm.BinaryOpr.Add:
                    ex = (ea + eb) if (ea is not None and eb is not None) else None
                    res = _with_exact_if_known(sign_add(sa, sb), ex)

                elif op == jvm.BinaryOpr.Sub:
                    ex = (ea - eb) if (ea is not None and eb is not None) else None
                    res = _with_exact_if_known(sign_sub(sa, sb), ex)

                elif op == jvm.BinaryOpr.Mul:
                    ex = (ea * eb) if (ea is not None and eb is not None) else None
                    res = _with_exact_if_known(sign_mul(sa, sb), ex)

                elif op == jvm.BinaryOpr.Div:
                    # definite divide-by-zero if exact divisor is 0, OR sign is definitely '0'
                    if (eb is not None and eb == 0) or sb == "0":
                        return "divide by zero"
                    ex = (ea // eb) if (ea is not None and eb is not None and eb != 0) else None
                    res = _with_exact_if_known(sign_div(sa, sb), ex)

                elif op == jvm.BinaryOpr.Rem:
                    if (eb is not None and eb == 0) or sb == "0":
                        return "divide by zero"
                    ex = (ea % eb) if (ea is not None and eb is not None and eb != 0) else None
                    res = _with_exact_if_known(sign_rem(sa, sb), ex)

                else:
                    res = ('int', 'TOP')

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


        case jvm.Get(field=f, static=True):
            # Resolve field name robustly (works with f.fieldid.name or f.name)
            field_name = None
            fid = getattr(f, "fieldid", None)
            if fid is not None:
                field_name = getattr(fid, "name", None)
            if field_name is None:
                field_name = getattr(f, "name", None)

            if field_name == "$assertionsDisabled":
                # Enable assertions: $assertionsDisabled == false (0)
                push(mk_int("0", 0))   # exact 0 so If/Ifz sees it precisely
                frame.pc += 1
                return [state]

            # Unknown static field → conservative TOP
            push(('int', 'TOP'))
            frame.pc += 1
            return [state]
    
        case jvm.Get(field=f):
            # Either mirror your concrete interpreter ("exception"),
            # or conservatively push TOP. Choose ONE to be consistent.
            return "exception"   # (or: push(('int','TOP')); frame.pc += 1; return [state])


        case jvm.Ifz(condition=cond, target=tgt):
            # Pop the tested value (integer compared against 0)
            v1 = pop()
            c = (cond or "").lower()   # 'eq','ne','lt','le','gt','ge'

            # Detect if the previous instruction was a DUP-family op (Dup, DupX1, ...)
            prev_was_dup = False
            try:
                prev = get_opcode(frame.pc.method, frame.pc.offset - 1)
                prev_was_dup = prev.__class__.__name__.lower().startswith("dup")
            except Exception:
                pass

            # ------------ helper: find the last 'iload <i>' producer ------------
            def _last_loaded_int_local(frame) -> int | None:
                off = frame.pc.offset
                for step in range(1, 8):  # look back up to 7 ops
                    idx = off - step
                    if idx < 0:
                        break
                    try:
                        instr = get_opcode(frame.pc.method, idx)
                    except Exception:
                        break

                    name = instr.__class__.__name__.lower()

                    # Skip DUP-family (value identical before the dup)
                    if name.startswith("dup"):
                        continue

                    # Direct 'iload i' → that local is being tested
                    if isinstance(instr, jvm.Load) and (isinstance(instr.type, jvm.Int) or instr.type == jvm.Int()):
                        return instr.index

                    # Constant int (iconst/ldc int) → not a local; stop the scan
                    if isinstance(instr, jvm.Push) and isinstance(getattr(instr, "value", None), int):
                        return None

                    # Arithmetic (e.g., isub) → value computed; don't refine a local
                    if isinstance(instr, jvm.Binary):
                        return None

                    # Unknown producer; stop conservatively
                    break
                return None
            # --------------------------------------------------------------------

            tested_local = _last_loaded_int_local(frame)

            # ---- decide truth value, preferring exact when available ----
            if isinstance(v1, tuple) and len(v1) >= 2 and v1[0] == 'int':
                ex = _exact(v1)  # may be None if not a known constant
                if ex is not None:
                    if   c == "eq": res = "true"  if ex == 0 else "false"
                    elif c == "ne": res = "false" if ex == 0 else "true"
                    elif c == "lt": res = "true"  if ex < 0  else "false"
                    elif c == "le": res = "true"  if ex <= 0 else "false"
                    elif c == "gt": res = "true"  if ex > 0  else "false"
                    elif c == "ge": res = "true"  if ex >= 0 else "false"
                    else:           res = "maybe"
                else:
                    # fall back to your sign predicate
                    res = unary_sign_cond_eval(v1[1], c)
            elif isinstance(v1, tuple) and v1[0] == 'ref':
                # Optional: handle null checks if your IR uses Ifz for refs too
                if c == "is":
                    res = "true" if v1[1] is None else "false"
                elif c == "isnot":
                    res = "false" if v1[1] is None else "true"
                else:
                    res = "maybe"
            else:
                res = "maybe"

            # ---- apply refinements & take the branch ----
            if res == "true":
                if tested_local is not None:
                    refine_local_in_state(state, tested_local, c, True)
                if prev_was_dup:
                    _refine_stack_top_in_state(state, c, True)
                frame.pc.offset = tgt
                return [state]

            if res == "false":
                if tested_local is not None:
                    refine_local_in_state(state, tested_local, c, False)
                if prev_was_dup:
                    _refine_stack_top_in_state(state, c, False)
                frame.pc += 1
                return [state]

            # res == "maybe" → fork & refine both paths
            s_true = State(
                frames=[Frame(locals=frame.locals.copy(),
                            stack=frame.stack.copy(),
                            pc=PC(frame.pc.method, tgt))],
                heap=state.heap.copy()
            )
            s_false = State(
                frames=[Frame(locals=frame.locals.copy(),
                            stack=frame.stack.copy(),
                            pc=PC(frame.pc.method, frame.pc.offset + 1))],
                heap=state.heap.copy()
            )

            if tested_local is not None:
                refine_local_in_state(s_true,  tested_local, c, True)
                refine_local_in_state(s_false, tested_local, c, False)
            if prev_was_dup:
                _refine_stack_top_in_state(s_true,  c, True)
                _refine_stack_top_in_state(s_false, c, False)

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
            c = (cond or "").lower()   # 'eq','ne','lt','le','gt','ge'

            # Helper: decide relation on exact ints
            def _decide_exact(a: int, b: int, c: str) -> str:
                if   c == 'eq': return 'true' if a == b else 'false'
                elif c == 'ne': return 'true' if a != b else 'false'
                elif c == 'lt': return 'true' if a <  b else 'false'
                elif c == 'le': return 'true' if a <= b else 'false'
                elif c == 'gt': return 'true' if a >  b else 'false'
                elif c == 'ge': return 'true' if a >= b else 'false'
                return 'maybe'

            # Helper: decide relation on abstract signs
            def _decide_signs(va, vb, c: str) -> str:
                # va, vb are ('int', Sign[, exact?]) tuples
                if not (_is_int(va) and _is_int(vb)):
                    return 'maybe'
                sa: Sign = va[1]  # type: ignore
                sb: Sign = vb[1]  # type: ignore
                return compare_two_signs(sa, sb, c)

            # Try to detect binary form: two ints on stack
            is_binary = False
            if len(frame.stack) >= 2:
                vtop = frame.stack[-1]
                v2   = frame.stack[-2]
                is_binary = (_is_int(vtop) and _is_int(v2))

            if is_binary:
                # ---- BINARY COMPARE: pop b then a; test a ? b ----
                b = pop()
                a = pop()

                # Prefer exact values when both known
                ea, eb = _exact(a), _exact(b)
                if ea is not None and eb is not None:
                    res = _decide_exact(ea, eb, c)
                else:
                    # fall back to sign relation
                    res = _decide_signs(a, b, c)

                # No local/dup refinements here (binary compare result doesn't single out one producer cleanly)
                if res == 'true':
                    frame.pc.offset = tgt
                    return [state]
                if res == 'false':
                    frame.pc += 1
                    return [state]

                # maybe → fork
                s_true = State(
                    frames=[Frame(locals=frame.locals.copy(),
                                stack=frame.stack.copy(),
                                pc=PC(frame.pc.method, tgt))],
                    heap=state.heap.copy()
                )
                s_false = State(
                    frames=[Frame(locals=frame.locals.copy(),
                                stack=frame.stack.copy(),
                                pc=PC(frame.pc.method, frame.pc.offset + 1))],
                    heap=state.heap.copy()
                )
                return [s_true, s_false]

            else:
                # ---- UNARY ZERO-COMPARE: pop v1; test v1 ? 0 ----
                v1 = pop()

                # Detect DUP-family (Dup, DupX1, ...) to refine the remaining copy
                prev_was_dup = False
                try:
                    prev = get_opcode(frame.pc.method, frame.pc.offset - 1)
                    prev_was_dup = prev.__class__.__name__.lower().startswith("dup")
                except Exception:
                    pass

                # Back-scan for a direct iload to refine that local
                def _last_loaded_int_local(frame) -> int | None:
                    off = frame.pc.offset
                    for step in range(1, 8):
                        idx = off - step
                        if idx < 0: break
                        try:
                            instr = get_opcode(frame.pc.method, idx)
                        except Exception:
                            break
                        name = instr.__class__.__name__.lower()
                        if name.startswith("dup"):
                            continue
                        if isinstance(instr, jvm.Load) and (isinstance(instr.type, jvm.Int) or instr.type == jvm.Int()):
                            return instr.index
                        if isinstance(instr, jvm.Push) and isinstance(getattr(instr, "value", None), int):
                            return None
                        if isinstance(instr, jvm.Binary):
                            return None
                        break
                    return None

                tested_local = _last_loaded_int_local(frame)

                # Decide truth preferring exact
                if _is_int(v1):
                    ex = _exact(v1)
                    if ex is not None:
                        res = _decide_exact(ex, 0, c)
                    else:
                        res = unary_sign_cond_eval(v1[1], c)  # type: ignore
                else:
                    res = 'maybe'

                # Apply refinements & branch
                if res == 'true':
                    if tested_local is not None:
                        refine_local_in_state(state, tested_local, c, True)
                    if prev_was_dup:
                        _refine_stack_top_in_state(state, c, True)
                    frame.pc.offset = tgt
                    return [state]

                if res == 'false':
                    if tested_local is not None:
                        refine_local_in_state(state, tested_local, c, False)
                    if prev_was_dup:
                        _refine_stack_top_in_state(state, c, False)
                    frame.pc += 1
                    return [state]

                # maybe → fork & refine
                s_true = State(
                    frames=[Frame(locals=frame.locals.copy(),
                                stack=frame.stack.copy(),
                                pc=PC(frame.pc.method, tgt))],
                    heap=state.heap.copy()
                )
                s_false = State(
                    frames=[Frame(locals=frame.locals.copy(),
                                stack=frame.stack.copy(),
                                pc=PC(frame.pc.method, frame.pc.offset + 1))],
                    heap=state.heap.copy()
                )
                if tested_local is not None:
                    refine_local_in_state(s_true,  tested_local, c, True)
                    refine_local_in_state(s_false, tested_local, c, False)
                if prev_was_dup:
                    _refine_stack_top_in_state(s_true,  c, True)
                    _refine_stack_top_in_state(s_false, c, False)
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
                push(('int', 'TOP'))
            frame.pc += 1
            return [state]

        case jvm.NewArray(type=elem_type, dim=dims):
            # pop length
            ln = pop()
            if ln[0] != 'int':
                return "negative array size"
            sign_len = ln[1]
            # if definitely negative -> error
            if sign_len == "-":
                return "negative array size"
            # determine concrete length only when integer input originally created array; for NewArray length is TOP/+, etc.
            # we can't create precise array with unknown length; approximate by length 0 if 0, otherwise 1 (small) to allow bounds checking when possible.
            if sign_len == "0":
                length = 0
            elif sign_len == "+":
                length = 1  # at least 1
            elif sign_len == "TOP":
                length = 1
            else:
                length = 0
            # choose element kind
            contains = getattr(elem_type, "contains", None)
            if contains == jvm.Char() or isinstance(elem_type, jvm.Char):
                arr = _ArrayObj('char', length, None)
            elif contains == jvm.Int() or isinstance(elem_type, jvm.Int):
                arr = _ArrayObj('int', length, None)
            else:
                arr = _ArrayObj('ref', length, None)
            oid = max(state.heap.keys()) + 1 if state.heap else 0
            state.heap[oid] = arr
            push(('ref', oid))
            frame.pc += 1
            return [state]

        case jvm.ArrayLoad(type=t):
            # pop index then arrayref (note concrete interpreter popped index then arrayref reverse)
            index = pop()
            arrref = pop()
            # null pointer?
            if arrref[0] != 'ref':
                return "null pointer"
            if arrref[1] is None:
                return "null pointer"
            oid = arrref[1]
            if oid not in state.heap:
                return "null pointer"
            arr_inst = state.heap[oid]
            # index analysis
            if index[0] != 'int':
                return "out of bounds"
            sidx = index[1]
            # definite negative -> out of bounds
            if sidx == "-":
                return "out of bounds"
            # definite zero -> check length
            if sidx == "0":
                if arr_inst.length <= 0:
                    return "out of bounds"
                elem = arr_inst.data[0]
                push(elem)
                frame.pc += 1
                return [state]
            # definite positive -> if length == 0 -> out of bounds else push TOP/unknown or first element
            if sidx == "+":
                if arr_inst.length == 0:
                    return "out of bounds"
                # can't decide which element -> use TOP element of kind
                push((arr_inst.elem_kind, "TOP"))
                frame.pc += 1
                return [state]
            # TOP index -> if array length == 0 then out of bounds, else push TOP
            if sidx == "TOP":
                if arr_inst.length == 0:
                    return "out of bounds"
                push((arr_inst.elem_kind, "TOP"))
                frame.pc += 1
                return [state]

        case jvm.ArrayLength():
            arr = pop()
            if arr[0] != 'ref':
                return "null pointer"
            ref = arr[1]
            if ref is None:
                return "null pointer"
            if ref not in state.heap:
                return "null pointer"
            length = state.heap[ref].length
            # push length as an abstract int
            if length == 0:
                push(('int', '0'))
            else:
                push(('int', '+'))  # any positive length we map to '+'
            frame.pc += 1
            return [state]

        case jvm.ArrayStore(type=t):
            value = pop()
            index = pop()
            arrref = pop()
            if arrref[0] != 'ref':
                return "null pointer"
            ref = arrref[1]
            if ref is None:
                return "null pointer"
            if ref not in state.heap:
                return "null pointer"
            arr_inst = state.heap[ref]
            if index[0] != 'int':
                return "out of bounds"
            sidx = index[1]
            if sidx == "-":
                return "out of bounds"
            # definite zero: store at 0 if length>0
            if sidx == "0":
                if arr_inst.length <= 0:
                    return "out of bounds"
                # coerce stored value to element kind
                if arr_inst.elem_kind == 'int':
                    if value[0] == 'int':
                        arr_inst.data[0] = ('int', value[1])
                    elif value[0] == 'char':
                        arr_inst.data[0] = ('int', value[1])
                    else:
                        arr_inst.data[0] = ('int', 'TOP')
                elif arr_inst.elem_kind == 'char':
                    if value[0] in ('char','int'):
                        arr_inst.data[0] = ('char', value[1])
                    else:
                        arr_inst.data[0] = ('char', 'TOP')
                else:
                    # ref
                    if value[0] == 'ref':
                        arr_inst.data[0] = ('ref', value[1])
                    else:
                        arr_inst.data[0] = ('ref', None)
                frame.pc += 1
                return [state]
            # positive or top index: if length==0 -> out of bounds else store TOP
            if sidx in ("+", "TOP"):
                if arr_inst.length == 0:
                    return "out of bounds"
                # store conservative TOP element
                if arr_inst.elem_kind == 'int':
                    arr_inst.data[0] = ('int', 'TOP')
                elif arr_inst.elem_kind == 'char':
                    arr_inst.data[0] = ('char', 'TOP')
                else:
                    arr_inst.data[0] = ('ref', None)
                frame.pc += 1
                return [state]

        case jvm.Incr(index=i, amount=d):
            # increment local i by constant d: update its sign accordingly
            val = frame.locals.get(i, ('int','TOP'))
            if val[0] != 'int':
                frame.locals[i] = ('int','TOP')
            else:
                # if definite 0 and d >0 => positive; if '-' and adding positive might be TOP etc.
                if val[1] == 'TOP':
                    frame.locals[i] = ('int','TOP')
                else:
                    # convert small constant d to sign and join
                    sd = sign_of_int(d)
                    frame.locals[i] = ('int', sign_add(val[1], sd))
            frame.pc += 1
            return [state]
        
        case jvm.Get(field=f, static=True):
            if getattr(f, "name", None) == "$assertionsDisabled":
                push(mk_int("+", 1))  # boolean true (assertions DISABLED by default)
                frame.pc += 1
                return [state]
            push(('int', 'TOP'))
            frame.pc += 1
            return [state]


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

# -------------------------
# Worklist interpreter
# -------------------------
def run_worklist(initial: State):
    worklist: List[State] = [initial]
    visited = set()  # to avoid infinite loops: store (method,offset,locals items) signatures

    while worklist:
        st = worklist.pop()
        # create a simple state key
        frame = st.frames[-1]
        key = (frame.pc.method, frame.pc.offset, tuple(sorted((k, v) for k,v in frame.locals.items())))
        if key in visited:
            continue
        visited.add(key)

        res = step_abstract(st)
        # If res is a string error, print and exit
        if isinstance(res, str):
            print(res)
            return
        # res is a list of next states
        for nxt in res:
            # special sentinel "ok" or "done"
            if isinstance(nxt, str):
                if nxt == "ok":
                    continue
                else:
                    # other strings -> print and exit
                    print(nxt)
                    return
            # check if nxt finished by empty frames -> treat as ok
            if not nxt.frames:
                continue
            # push next into worklist
            worklist.append(nxt)
    # no errors found on any path -> ok
    print("ok")

# run
run_worklist(initial_state)
