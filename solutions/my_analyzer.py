import sys
import re

import logging
import jpamb

from tree_sitter import Language, Parser
from tree_sitter import Node
import tree_sitter_java as tsjava

def text(node):  # convenience
    return my_bytes[node.start_byte:node.end_byte].decode("utf-8")

def walk(node):
    yield node
    for c in node.children:
        yield from walk(c)

def node_to_sexp(node, my_bytes, *, named_only=False):
    """Return an s-expression string for a tree_sitter.Node."""
    kids = node.named_children if named_only else node.children
    txt = my_bytes[node.start_byte:node.end_byte].decode("utf-8")
    if not kids:
        return node.type + f' "{txt}"'
    return f"({node.type} {txt} " + " ".join(node_to_sexp(c, my_bytes, named_only=named_only) for c in kids) + ")"
    


def find_method_open_brace(src: str, name: str) -> int:
    sig = re.compile(
        rf'(?<!\.)\b{re.escape(name)}\s*\([^)]*\)\s*(?:throws\s+[\w\.,\s<>?\[\]]+)?\s*\{{',
        #rf'{name}.*\{{',
        re.DOTALL
    )
    m = sig.search(src)
    return m.end() - 1 if m else -1

def slice_balanced_block(src: str, open_idx: int) -> str | None:
    if open_idx < 0 or src[open_idx] != '{': return None
    i, n, depth = open_idx, len(src), 0
    in_sl = in_ml = in_str = in_chr = False
    esc = False
    start = open_idx + 1
    while i < n:
        ch = src[i]; nxt = src[i+1] if i+1 < n else ''
        if in_sl:
            if ch == '\n': in_sl = False
        elif in_ml:
            if ch == '*' and nxt == '/': in_ml = False; i += 1
        elif in_str:
            if not esc and ch == '"': in_str = False
            esc = not esc and ch == '\\'
        elif in_chr:
            if not esc and ch == "'": in_chr = False
            esc = not esc and ch == '\\'
        else:
            if ch == '/' and nxt == '/': in_sl = True; i += 1
            elif ch == '/' and nxt == '*': in_ml = True; i += 1
            elif ch == '"': in_str = True; esc = False
            elif ch == "'": in_chr = True; esc = False
            elif ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0: return src[start:i]
        i += 1
    return None

log = logging
log.basicConfig(level=logging.DEBUG)

methodid = jpamb.getmethodid(
    "mixed effort Analyzer",
    "1.0 beta",
    "Shit on wheels",
    ["mixed", "python"],
    for_science=True,
)

if len(sys.argv) == 2 and sys.argv[1] == "info":
    # Output the 5 required info lines
    print("mixed effort Analyzer")
    print("1.0 beta")
    print("Shit on wheels") #"Student Group Name"
    print("mixed,python")
    print("no")  # Use any other string to share system info
else:

    has_assert = False
    has_assert_in_bytecode = False
    has_div_by_zero = False
    # --- Heuristic probabilities ---
    ok = 50
    div0 = 50
    asrt = 50
    oob = 50
    npe = 50
    inf = 50

    # Get the method we need to analyze
    m = jpamb.Suite().findmethod(methodid)
    log.debug(m)
    
    for inst in m["code"]["bytecode"]:
        if (
            inst["opr"] == "invoke"
            and inst["method"]["ref"]["name"] == "java/lang/AssertionError"
        ):
            has_assert_in_bytecode = True
            break
    else:
        # I'm pretty sure the answer is no
        has_assert_in_bytecode = False
    
    classname, methodname, args = re.match(r"(.*)\.(.*):(.*)", sys.argv[1]).groups()
    
    log.debug(sys.argv)
    log.debug(classname)
    log.debug(methodname)
    log.debug(args)
    
    #open the class file to read
    path = "./src/main/java/" + classname.replace(".","/") + ".java"
    with open(path, "r") as f:
        data = f.read()
        
        #pattern = rf"\b{re.escape(methodname)}\s*\([^)]*\)\s*\{{(.*?)\}}"
        #m = re.search(pattern, data, flags=re.DOTALL)
        #func = m.group(1) if m else None
        #print(func)
        
        #Get the method from the methodname body
        open_idx = find_method_open_brace(data, methodname)
        body = slice_balanced_block(data, open_idx)

        try:
            JAVA_LANGUAGE = Language(tsjava.language())
        except AttributeError:
            JAVA_LANGUAGE = tsjava.LANGUAGE

        parser = Parser(JAVA_LANGUAGE)              

        my_bytes = body.encode()
        tree = parser.parse(my_bytes)
        root = tree.root_node

        #TREE
        log.debug(node_to_sexp(root, my_bytes))

        for n in walk(root):
            # Detect an assert statement
            if n.type == "assert_statement":
                has_assert = True

            # Detect "X / 0" (any numeric expression dividing by the literal 0)
            if n.type == "binary_expression" and text(n).find("/") != -1:
                # Tree-sitter Java structure: left operand, operator, right operand
                # We'll conservatively check the source slice for "/ 0" or "/0"
                src = text(n)
                # Very simple literal-0 check; could be improved to inspect child nodes
                if re.search(r"/\s*0(?![0-9])", src):
                    has_div_by_zero = True
        
        #The Method
        log.debug(f"{body}")

        if has_div_by_zero and has_assert:
            # Two possible failures; never ok.
            # Typical runs (assertions disabled) hit divide-by-zero.
            div0 = 85
            asrt = 10
        elif has_div_by_zero:
            div0 = 85
        elif has_assert and has_assert_in_bytecode:
            asrt = 80
            ok = 20
        elif has_assert:
            # With only an assert (no div-by-zero), we can't know b.
            # Split between ok and assertion error.
            asrt = 50
            ok = 50
        else:
            ok = 60
        
    # Make predictions (improve these by looking at the Java code!)
    ok_chance = f"{ok}%"
    divide_by_zero_chance = f"{div0}%"
    assertion_error_chance = f"{asrt}%"
    out_of_bounds_chance = f"{oob}%"
    null_pointer_chance = f"{npe}%"
    infinite_loop_chance = f"{inf}%"
    
    if has_assert_in_bytecode:
        log.debug("has_assert_in_bytecode = True")
    
    # Output predictions for all 6 possible outcomes
    print(f"ok;{ok_chance}")
    print(f"divide by zero;{divide_by_zero_chance}") 
    print(f"assertion error;{assertion_error_chance}")
    print(f"out of bounds;{out_of_bounds_chance}")
    print(f"null pointer;{null_pointer_chance}")
    print(f"*;{infinite_loop_chance}")
