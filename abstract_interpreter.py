from dataclasses import dataclass
from typing import Literal, TypeAlias, Union, Optional, List, Dict, Tuple
import jpamb
from jpamb import jvm

# -------------------------
# Sign lattice definitions
# -------------------------
Sign: TypeAlias = Literal["+", "-", "0", "TOP", "BOT"]

def sign_of_int(x: int) -> Sign:
    if x > 0:
        return "+"
    if x < 0:
        return "-"
    return "0"

def sign_join(a: Sign, b: Sign) -> Sign:
    if a == b:
        return a
    if a == "BOT":
        return b
    if b == "BOT":
        return a
    return "TOP"

def sign_add(a: Sign, b: Sign) -> Sign:
    if a == "BOT" or b == "BOT": return "BOT"
    if a == "TOP" or b == "TOP": return "TOP"
    if a == "0": return b
    if b == "0": return a
    if a == b: return a
    if a == "-" and b == "-": return "-"
    if a == "+" and b == "+": return "+"
    return "TOP"

def sign_sub(a: Sign, b: Sign) -> Sign:
    if a == "BOT" or b == "BOT": return "BOT"
    if a == "TOP" or b == "TOP": return "TOP"
    if a == "0" and b == "0": return "0"
    if a == "0": return "TOP"
    if b == "0": return a
    if a == b: return "TOP"
    if a == "+" and b == "-": return "+"
    if a == "-" and b == "+": return "-"
    return "TOP"

def sign_mul(a: Sign, b: Sign) -> Sign:
    if a == "BOT" or b == "BOT": return "BOT"
    if a == "TOP" or b == "TOP": return "TOP"
    if a == "0" or b == "0": return "0"
    return "+" if a == b else "-"

def sign_div(a: Sign, b: Sign) -> Sign:
    # division by zero -> BOT (error) handled outside
    if a == "BOT" or b == "BOT": return "BOT"
    if a == "0": return "0"
    #if b == "0": return "division by zero"  # should not happen; division by zero handled outside
    if a == "TOP" or b == "TOP": return "TOP"
    return "+" if a == b else "-"

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
def unary_sign_cond_eval(sign: Sign, cond: str) -> str:
    # cond: "eq", "ne", "lt", "le", "gt", "ge", "is", "isnot"
    if cond == "eq":
        if sign == "0": return "true"
        if sign in {"+", "-"}: return "false"
        return "maybe"
    if cond == "ne":
        if sign == "0": return "false"
        if sign in {"+", "-"}: return "true"
        return "maybe"
    if cond == "lt":
        if sign == "-": return "true"
        if sign in {"0", "+"}: return "false"
        return "maybe"
    if cond == "le":
        if sign in {"-", "0"}: return "true"
        if sign == "+": return "false"
        return "maybe"
    if cond == "gt":
        if sign == "+": return "true"
        if sign in {"0", "-"}: return "false"
        return "maybe"
    if cond == "ge":
        if sign in {"+", "0"}: return "true"
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
        init_locals[i] = ('int', sign_of_int(v.value))
    elif isinstance(v.type, jvm.Boolean) or v.type == jvm.Boolean():
        # booleans are ints 0/1
        init_locals[i] = ('int', "0" if v.value == 0 else "+")
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
                push(('int', sign_of_int(v.value)))
            elif isinstance(v.type, jvm.Boolean) or v.type == jvm.Boolean():
                push(('int', "0" if v.value == 0 else "+"))
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
            print("DEBUG a and b: Dividing", a, b)
            if a[0] != 'int' or b[0] != 'int':
                # if not ints, produce TOP
                print("Setting TOP due to non-int operand")
                res = ('int', 'TOP')
            else:
                sa: Sign = a[1]  # type: ignore
                sb: Sign = b[1]  # type: ignore
                if op == jvm.BinaryOpr.Add:
                    res = ('int', sign_add(sa, sb))
                elif op == jvm.BinaryOpr.Sub:
                    # Note: concrete did pop v2, v1 then v1 - v2; we popped in same order
                    res = ('int', sign_sub(sa, sb))
                elif op == jvm.BinaryOpr.Mul:
                    res = ('int', sign_mul(sa, sb))
                elif op == jvm.BinaryOpr.Div:
                    # if divisor definitely zero -> error
                    print("DEBUG: Dividing", sa, sb)
                    if sb == "0":
                        return "divide by zero"
                    if sb == "TOP":
                        print("possible divide by zero")
                    # else proceed and push abstract result
                    res = ('int', sign_div(sa, sb))
                elif op == jvm.BinaryOpr.Rem:
                    if sb == "0":
                        return "divide by zero"
                    # remainder sign is hard to compute precisely; approximate:
                    if sa == "0":
                        res = ('int', '0')
                    else:
                        res = ('int', 'TOP')
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

        case jvm.Get(static=True):
            # handle $assertionsDisabled style read
            name = op.field.fieldid.name
            if name == "$assertionsDisabled":
                push(('int', '0' if True else '1'))  # we assume assertions enabled -> push boolean 0
                frame.pc += 1
                return [state]
            # fallback: push 0
            push(('int', '0'))
            frame.pc += 1
            return [state]

        case jvm.Ifz(condition=cond, target=tgt):
            v1 = pop()
            # cond is a str like 'eq','ne','lt', etc.
            c = (cond or "").lower()
            if v1[0] == 'int':
                res = unary_sign_cond_eval(v1[1], c)
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
                res = compare_two_signs(v1[1], v2[1], c)
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

def run_worklist(initial: State):
    worklist: List[State] = [initial]
    visited = set()
    reached_exit = False
    reached_error: Optional[str] = None

    while worklist:
        st = worklist.pop()
        if not st.frames:
            reached_exit = True
            continue

        frame = st.frames[-1]
        key = (frame.pc.method, frame.pc.offset, frozenset(frame.locals.items()))
        if key in visited:
            continue
        visited.add(key)

        res = step_abstract(st)

        # If step_abstract reports a string, it's an error
        if isinstance(res, str):
            reached_error = res
            break

        for nxt in res:
            if isinstance(nxt, str):
                if nxt == "ok":
                    reached_exit = True
                else:
                    reached_error = nxt
                    break
            elif not nxt.frames:
                reached_exit = True
            else:
                worklist.append(nxt)

        if reached_error:
            break

    # --- decide final verdict ---
    if reached_error:
        print(reached_error)
    elif reached_exit:
        print("ok")
    else:
        # no exit, no error -> infinite loop
        print("*")

# run
run_worklist(initial_state)
