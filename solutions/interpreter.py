import jpamb
from jpamb import jvm
from dataclasses import dataclass

import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="[{level}] {message}")


ASSERTIONS_ENABLED = True




methodid, input = jpamb.getcase()

@dataclass
class PC:
    method: jvm.AbsMethodID
    offset: int

    def __iadd__(self, delta):
        self.offset += delta
        return self

    def __add__(self, delta):
        return PC(self.method, self.offset + delta)

    def __str__(self):
        return f"{self.method}:{self.offset}"


#@dataclass
#class Bytecode:
#    suite: jpamb.Suite
#    methods: dict[jvm.AbsMethodID, list[jvm.Opcode]]
#
#    def __getitem__(self, pc: PC) -> jvm.Opcode:
#        try:
#            opcodes = self.methods[pc.method]
#        except KeyError:
#            opcodes = list(self.suite.method_opcodes(pc.method))
#            self.methods[pc.method] = opcodes
#
#        return opcodes[pc.offset]
    
@dataclass
class Bytecode:
    suite: jpamb.Suite
    methods: dict[jvm.AbsMethodID, list[jvm.Opcode]]
    # add this:
    offset_to_index: dict[jvm.AbsMethodID, dict[int, int]] = None

    def __post_init__(self):
        if self.offset_to_index is None:
            self.offset_to_index = {}

    def _ensure_loaded(self, method: jvm.AbsMethodID):
        if method not in self.methods:
            ops = list(self.suite.method_opcodes(method))
            self.methods[method] = ops
            # build a precise offset→index map
            self.offset_to_index[method] = {op.offset: i for i, op in enumerate(ops)}

    def __getitem__(self, pc: PC) -> jvm.Opcode:
        self._ensure_loaded(pc.method)
        return self.methods[pc.method][pc.offset]


@dataclass
class Stack[T]:
    items: list[T]

    def __bool__(self) -> bool:
        return len(self.items) > 0

    @classmethod
    def empty(cls):
        return cls([])

    def peek(self) -> T:
        return self.items[-1]

    def pop(self) -> T:
        return self.items.pop(-1)

    def push(self, value):
        self.items.append(value)
        return self

    def __str__(self):
        if not self:
            return "ϵ"
        return "".join(f"{v}" for v in self.items)


suite = jpamb.Suite()
bc = Bytecode(suite, dict())


@dataclass
class Frame:
    locals: dict[int, jvm.Value]
    stack: Stack[jvm.Value]
    pc: PC

    def __str__(self):
        locals = ", ".join(f"{k}:{v}" for k, v in sorted(self.locals.items()))
        return f"<{{{locals}}}, {self.stack}, {self.pc}>"

    def from_method(method: jvm.AbsMethodID) -> "Frame":
        return Frame({}, Stack.empty(), PC(method, 0))

# Tiny object handle we can tag with a class name
#class _ObjRef:
#    __slots__ = ("class_name",)
#    def __init__(self, class_name: str):
#        self.class_name = class_name

@dataclass
class State:
    heap: dict[int, jvm.Value]
    frames: Stack[Frame]

    def __str__(self):
        return f"{self.heap} {self.frames}"
    
@dataclass
class _ArrayObj:
    __slots__ = ("type", "length", "data")
    def __init__(self, elem_type, length):
        self.type = elem_type
        self.length = length
        # default-fill by element kind
        if isinstance(elem_type, jvm.Int) or elem_type == jvm.Int():
            self.data = [jvm.Value.int(0) for _ in range(length)]
        elif isinstance(elem_type, jvm.Char) or elem_type == jvm.Char():
            self.data = [jvm.Value.char('\x00') for _ in range(length)]
        else:
            # reference / other kinds start null
            self.data = [None for _ in range(length)]

@dataclass
class _ObjRef:
    __slots__ = ("class_name", "inited")
    def __init__(self, class_name: str):
        self.class_name = str(class_name)
        self.inited = False  # set true by <init
'''
def _jump_pc(method: jvm.AbsMethodID, tgt_offset: int) -> PC:
    # Load bytecodes & precompute map if needed
    bc._ensure_loaded(method)
    idx_map = bc.offset_to_index[method]
    try:
        logger.debug(f"Jumping to offset {tgt_offset} in {method}, idx_map: {idx_map}") 
        idx = idx_map[tgt_offset]
    except KeyError:
        raise ValueError(f"Branch target offset {tgt_offset} not found in {method}")
    return PC(method, idx)
'''
def _jump_pc(method: jvm.AbsMethodID, tgt: int) -> PC:
    # ensure ops loaded
    ops = bc.methods.get(method)
    if ops is None:
        ops = list(bc.suite.method_opcodes(method))
        bc.methods[method] = ops

    n = len(ops)

    # Heuristic 1: if tgt is a valid opcode index and doesn't equal that op's offset,
    # treat it as an index jump (common in simple methods).
    if 0 <= tgt < n and ops[tgt].offset != tgt:
        return PC(method, tgt)

    # Heuristic 2: try bytecode offset match (common in bigger methods).
    for i, op in enumerate(ops):
        if op.offset == tgt:
            return PC(method, i)

    # Heuristic 3: if still ambiguous but index is in-range, use index.
    if 0 <= tgt < n:
        return PC(method, tgt)

    raise ValueError(f"Branch target {tgt} not found in {method}")

def _is_void_type(t) -> bool:
    # Works with multiple possible encodings of void
    if t is None:                      # sometimes void is encoded as None
        return True
    # many bytecode APIs carry a JVM descriptor
    if getattr(t, "descriptor", None) == "V":
        return True
    # some expose a printable form
    s = str(t).lower()
    return s in {"v", "void", "returnvoid"}

def step(state: State) -> State | str:
    #assert isinstance(state, State), f"expected frame but got {state}"
    if isinstance(state, str):
        # If the program is already finished (state is an error string or "ok"), 
        # return the result immediately to prevent crashing on the next line.
        return state
    
    frame = state.frames.peek()
    opr = bc[frame.pc]
    logger.debug(f"STEP {opr}\n{state}")
    match opr:
        case jvm.Push(value=v):
            frame.stack.push(v)
            frame.pc += 1
            return state
        case jvm.Load(type=jvm.Int(), index=i):
            logger.debug(frame.locals)
            frame.stack.push(frame.locals[i])
            frame.pc += 1
            return state
        case jvm.Binary(type=jvm.Int(), operant=o): #jvm.BinaryOpr.Div
            v2, v1 = frame.stack.pop(), frame.stack.pop()
            assert v1.type is jvm.Int(), f"expected int, but got {v1}"
            assert v2.type is jvm.Int(), f"expected int, but got {v2}"
            
            match o:
                case jvm.BinaryOpr.Add: 
                    frame.stack.push(jvm.Value.int(v1.value + v2.value))
                case jvm.BinaryOpr.Sub: 
                    frame.stack.push(jvm.Value.int(v1.value - v2.value))
                case jvm.BinaryOpr.Mul: 
                    frame.stack.push(jvm.Value.int(v1.value * v2.value))
                case jvm.BinaryOpr.Div: 
                    if v2.value == 0:
                        return "divide by zero"
                    frame.stack.push(jvm.Value.int(v1.value // v2.value))
                case jvm.BinaryOpr.Rem: 
                    if v2.value == 0:
                        return "divide by zero"
                    frame.stack.push(jvm.Value.int(v1.value % v2.value))
                case _: 
                    raise NotImplementedError(f"Unhandled binary operant: {o}")

            frame.pc += 1
            return state
        case jvm.Return(type=type):
            if type is not None: 
                v1 = frame.stack.pop()
                assert v1.type == type, f"expected {type}, but got {v1}"
            
            state.frames.pop()
            if state.frames:
                frame = state.frames.peek()
                if type is not None:
                    frame.stack.push(v1)
                frame.pc += 1
                return state
            else:
                return "ok"
        case jvm.Cast(from_=jvm.Int(), to_=jvm.Short()):
            v1 = frame.stack.pop()
            assert v1.type == jvm.Int(), f"expected type int, but got {v1}"
            short_val = ((v1.value + 2*15) % 216) - 2*15 
            frame.stack.push(jvm.Value(type=jvm.Int(), value=short_val))
            frame.pc += 1
            return state     
        case jvm.If(condition=cond, target=t):
            v2 = frame.stack.pop()
            v1 = frame.stack.pop()
            c = (cond or "").lower()
            if isinstance(v1.type, jvm.Int) and isinstance(v2.type, jvm.Int):
                if c == "eq":
                     take =  (v1.value == v2.value)
                elif c == "ne":
                    take =  (v1.value != v2.value)
                elif c == "lt":
                    take =  (v1.value < v2.value)
                elif c == "le":
                    take =  (v1.value <= v2.value)
                elif c == "gt":
                    take =  (v1.value > v2.value)
                elif c == "ge":
                    take =  (v1.value >= v2.value)
                else:
                    raise NotImplementedError(f"Unknown If condition: {cond!r}")
            elif isinstance(v1.type, jvm.Reference) and isinstance(v2.type, jvm.Reference):
                if c == "is":
                    take = (v1 == v2)
                elif c == "isnot":
                    take = (v1 != v2)
                else:
                    raise NotImplementedError(f"Unknown If condition: {cond!r}")
            else:
                raise TypeError(f"if expected two ints or two refs, got {v1}, {v2}")
            if take:
                frame.pc.offset = t
            else:
                frame.pc += 1
            return state  
        case jvm.Ifz(condition=cond, target=tgt):
            v1 = frame.stack.pop()
            cond = (cond or "").lower()
            if isinstance(v1.type, jvm.Int):
                match cond:
                    case "eq":
                        take = v1.value == 0
                    case "ne":
                        take = v1.value != 0
                    case "lt":
                        take = v1.value < 0
                    case "le":
                        take = v1.value <= 0
                    case "gt":
                        take = v1.value > 0
                    case "ge":
                        take = v1.value >= 0
                    case _:
                        raise NotImplementedError(f"Don't know how to handle the condition: {opr!r}")    
            elif isinstance(v1.type, jvm.Reference):
                if cond == "is":
                    take = (v1 == 0)
                elif c == "isnot":
                    take = (v1 != 0)
            else:
                raise TypeError(f"ifz expected int or ref, but got {v1}")
        
            if take:
                frame.pc.offset = tgt
            else:
                frame.pc += 1
            return state   
        case jvm.Goto(target=tgt):
            frame.pc = PC(frame.pc.method, tgt) 
            #frame.pc = _jump_pc(frame.pc.method, tgt)
            return state
        case jvm.Load(type=t, index=i):
            frame.stack.push(frame.locals[i])
            frame.pc += 1
            return state
        case jvm.Store(type=t, index=i):
            # pop value to store
            val = frame.stack.pop()

            if isinstance(t, jvm.Reference):
                # accept any object as a reference; don’t assert identity
                frame.locals[i] = val
                frame.pc += 1
                return state

            assert val.type is t, f"expected {t}, but got {val}"
            # coerce to expected type (keeps locals consistent)
            if t is jvm.Int():
                val = jvm.Value.int(val.value)
            elif t is jvm.Boolean():
                val = jvm.Value.int(1 if val.value != 0 else 0)
            elif t is _ArrayObj():
                val = val  # already an _ArrayObj
            else:
                assert False, f"not implemented for type {t}"
            # write local
            frame.locals[i] = val

            frame.pc += 1
            return state
        case jvm.ArrayLoad(type=t):
            # Stack: ..., arrayref, index  → pop in reverse order
            index = frame.stack.pop().value
            arr = frame.stack.pop()

            ref = arr.value
            if ref is None:
                return "null pointer"

            instance = state.heap[ref]

            if not isinstance(instance, _ArrayObj):
                return "null pointer"
            if index < 0:
                return "out of bounds"
            if index >= instance.length:
                return "out of bounds"
            
            elem = instance.data[index]
        
            if t is jvm.Char():
                frame.stack.push(jvm.Value.int(ord(elem.value)))
                logger.debug(f"push Char: {elem.value}" + "Its type is Char")
                #frame.stack.push(jvm.Value.char(instance.data[index].value))
            elif t is jvm.Int():
                frame.stack.push(jvm.Value.int(elem.value))
                logger.debug(f"push INT: {elem.value}" + "Its type is Int")
            else:
                frame.stack.push(elem)
                logger.debug(f"push Ref: {elem.value}" + "Its type is Ref")

            frame.pc += 1
            return state

        case jvm.ArrayLength():
            arr = frame.stack.pop()

            ref = getattr(arr, "value", arr)
            if ref is None:
                return "null pointer"
            
            logger.debug(f"ArrayIndex: {ref}")
            instance = state.heap[ref]

            if not isinstance(instance, _ArrayObj):
                return "null pointer"
                #raise TypeError(f"Expected _ArrayObj From ArrayLength, got {type(arr)}")
            frame.stack.push(jvm.Value.int(instance.length))
            frame.pc += 1
            return state

        case jvm.ArrayStore(type=t):
            # Stack: ..., arrayref, index, value  → pop in reverse order
            value = frame.stack.pop()
            index = frame.stack.pop().value
            arr   = frame.stack.pop().value


            if arr is None:
                return "null pointer" 
            
            instance = state.heap[arr]
  
            if not isinstance(instance, _ArrayObj):
                return "null pointer"
                #raise TypeError(f"Expected _ArrayObj in ArrayStore, got {type(instance)}")
            if index < 0 or index >= instance.length:
                return "out of bounds"
            
            logger.debug(f"Storing {value} of type {t} at index {index} of array {instance} of type {instance.type}")
            
            elem_t = instance.type

            if isinstance(elem_t, jvm.Char):
                value = jvm.Value.int(ord(value.value))
            elif isinstance(elem_t, jvm.Int):
                value = jvm.Value.int(value.value)
            else:
                value = value  # reference or other type; accept as-is
            
            instance.data[index] = value
            frame.pc += 1
            return state  

        case jvm.NewArray( type=elem_type, dim=dims):
            # Array length is on the stack):
            length = frame.stack.pop()
            if length.value < 0:
                return "negative array size"
            arr_type = elem_type # jvm.Array(elem_type)
            arr_obj = _ArrayObj(arr_type, length.value)
            idx = len(state.heap)
            state.heap[idx] = arr_obj

            frame.stack.push(jvm.Value.int(idx))
            
            #frame.stack.push(arr_obj) # a Value of array type
            frame.pc += 1
            return state
        
        case jvm.InvokeVirtual(method=m, is_interface=_):
            #cls, name, arg_types, ret = _minfo(m)
            assert False, "InvokeVirtual not implemented yet" + "This is m:"+repr(m)

        case jvm.InvokeSpecial(method=m, is_interface=_):
            #cls, name, arg_types, ret = _minfo(m)
            #arg_types = m.get("args", []) or []
            #args_list = list(getattr(m, "args", []) or [])
            args_list = m.extension.params._elements

            args = [frame.stack.pop() for _ in range(len(args_list))][::-1]
            index_objref = frame.stack.pop()  # consume the receiver
            obj = state.heap.get(index_objref.value)

            #raise NotImplementedError("InvokeSpecial not implemented yet" + "This is m:"+repr(m) + "This is obj:"+repr(obj) + "WTF:")# + obj.get("class"))
            if m.classname.name == "java/lang/Object" and m.methodid.name == "<init>":
                # Object.<init> is a no-op
                frame.pc += 1
                return state
            elif m.classname.name == "java/lang/AssertionError" and m.methodid.name == "<init>":  
                # AssertionError.<init> is a no-op
                frame.pc += 1
                return state
            
            
            newframe = Frame.from_method(m)

            state.frames.push(newframe)
            
            return state
            #raise NotImplementedError("InvokeSpecial not implemented yet" + "This is m:"+repr(m) + "This is obj:"+repr(obj))

        case jvm.InvokeStatic(method=m):
                # 1) Determine arity from the method signature object
            #    (jpamb encodes params on m.extension.params)
            param_types = list(m.extension.params or [])
            argc = len(param_types)

            # 2) Pop arguments in reverse, then restore order (left->right)
            args = [frame.stack.pop() for _ in range(argc)]
            args.reverse()

            # 3) (Optional) light coercions to keep locals consistent
            #    Only needed if your stack can contain e.g. char/bool you want normalized to Int
            norm_args: list[jvm.Value] = []
            for t, v in zip(param_types, args):
                norm_args.append(v)  # default: no coercion
                #match t:
                #    case jvm.Int() | jvm.Boolean() | jvm.Byte() | jvm.Short() | jvm.Char():
                #        # normalize everything int-like to Value.int(...)
                #        # (If v is already int Value, this is a no-op)
                #        if t is jvm.Char():
                #            v = jvm.Value.int(ord(v.value))
                #        else:
                #            v = jvm.Value.int(v.value)
                #        norm_args.append(v)
                #    case _:
                #        norm_args.append(v.value)

            # 4) Create callee frame with locals 0..n-1 filled with args
            callee = Frame.from_method(m)
            callee.locals = {i: v for i, v in enumerate(norm_args)}

            #bc._ensure_loaded(m)
            #if not bc.methods[m]:
            #    raise NotImplementedError(f"Method {m} has no bytecodes")

            #if m not in bc.offset_to_index:
            #    bc.offset_to_index[m] = {}

            #first_value = next(iter(bc.offset_to_index[m].values()))

            #callee.pc.offset = first_value


            callee.pc = PC(callee.pc.method, callee.pc.offset) 
            
            # 5) Push the new frame; execution will continue at m:0
            state.frames.push(callee)
            return state
                
        case jvm.Throw():

            index_objref = frame.stack.pop()  # consume the receiver
            obj = state.heap.get(index_objref.value)
            
            #raise NotImplementedError("Throw not implemented yet" + "This is obj:"+repr(obj))
            if obj is None:
                return "exception" 

            if obj["class"].name == "java/lang/AssertionError":
                return "assertion error"
            return "exception"

        case jvm.New(classname=cls):
            #frame.stack.push(_ObjRef(str(cls)))
            heap = state.heap
            idx = len(heap)
            heap[idx] = {"class": cls, "fields": {}}
            frame.stack.push(jvm.Value.int(idx))

            frame.pc += 1
            return state
  
        case jvm.Dup():
            v = frame.stack.peek()
            frame.stack.push(v)
            frame.pc += 1   
            return state
        
        case jvm.Incr(index=i, amount=d):
            v = frame.locals[i]
            assert v.type is jvm.Int(), f"expected int, but got {v}"
            v = jvm.Value.int(v.value + d)
            frame.locals[i] = v
            frame.pc += 1
            return state

        case jvm.Get(field=f):
            # Field metadata (objects, not dicts)
            fld = opr.field                  # AbsFieldID(classname=..., extension=FieldID(name=..., type=...))
            t   = fld.extension.type         # e.g., jvm.Boolean(), jvm.Int(), ...
            name = fld.extension.name
            cls  = getattr(fld, "classname", "")  # e.g., "jpamb/cases/Calls"

            # ---------- STATIC FIELD READ ----------
            if opr.static:
                if name == "$assertionsDisabled":
                    # Force 'assert' disabled for these classes (push boolean true == 0)
                    #if cls in ("jpamb/cases/Calls", "jpamb/cases/Arrays"):
                    frame.stack.push(jvm.Value.int(0 if ASSERTIONS_ENABLED else 1))
                    frame.pc += 1
                    return state

                    # Fallback: honor your global switch, if you want to keep it
                    # (Also: don't rely on identity equality for the type.)
                    #enabled = bool(globals().get("ASSERTIONS_ENABLED", True))
                
                frame.stack.push(jvm.Value.int(0))
                frame.pc += 1
                return state
            # Instance fields not implemented yet
            objref = frame.stack.pop()
            assert False, "Get instance field not implemented"

        case a:
            a.help()
            raise NotImplementedError(f"Don't know how to handle: {a!r}")

frame = Frame.from_method(methodid)
heap: dict[int, jvm.Value] = {}

for i, v in enumerate(input.values):
    logger.debug(v)
    match v.type:
        case jvm.Boolean():
             frame.locals[i] = jvm.Value.int(1 if v.value else 0)
        case jvm.Int():
             frame.locals[i] = v
        case jvm.Char():
            frame.locals[i] = jvm.Value.int(ord(v.value))
        case jvm.Array():
            # v.value might be chars ('a') or code points (97); normalize to chars
            logger.warning("Array input values not implemented ....value " + repr(v.value) + " at index " + repr(i) + " of " + repr(input.values))
            
            if(v.type.contains == jvm.Char()):
                logger.debug("Array of Char detected")
                elems = [jvm.Value.int(x) for x in (v.value or [])]     
                arr_inst = _ArrayObj(elem_type=jvm.Char(), length=len(elems))
                arr_inst.data = elems
                oid = len(heap)
                heap[oid] = arr_inst
                frame.locals[i] = jvm.Value.int(oid)

            elif(v.type.contains == jvm.Int()):
                elems = [jvm.Value.int(x) for x in (v.value or [])]     
                arr_inst = _ArrayObj(elem_type=jvm.Int(), length=len(elems))
                arr_inst.data = elems
                oid = len(heap)
                heap[oid] = arr_inst
                frame.locals[i] = jvm.Value.int(oid)
            else:
                logger.debug("Array of unknown type detected" + repr(v.type))
                assert False, "CRAP"
            
        case _:
            logger.warning(f"Unhandled input value type: {v.type} and value {v.value} at index {i} of {input.values}")
            assert False, repr(v)

state = State(heap, Stack.empty().push(frame))


# constant for infiite loop detection, if exceeded then infinite loop is likely the case.
MAX_INSTRUCTIONS = 10_000_000
instruction_count = 0

# infinite loop detection
while isinstance(state, State):
    instruction_count += 1
    if instruction_count > MAX_INSTRUCTIONS:
        print("Infinite loop detected")
        exit() 
    state = step(state) 
    
# If the loop breaks, 'state' will hold result of execution - either "ok" or "out of bounds" etc
print(state)





for x in range(1000000):
    state = step(state)
    if isinstance(state, str):
        print(state)
        break
    else:
        print("*")
        
        

