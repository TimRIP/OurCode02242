import sys
import re

import logging
import jpamb

from abstract import analyze_method_no_inputs
from abstract import get_abstract_warnings

# Always present
methodid = jpamb.parse_methodid(sys.argv[1])

classname, methodname, args = re.match(r"(.*)\.(.*):(.*)", sys.argv[1]).groups()

logging.error(args)
# Try to get concrete inputs if present (interpret mode); else run with TOPs
abstract_outcome = analyze_method_no_inputs(methodid)
warnings = get_abstract_warnings()
# Use abstract_outcome in your analyzer logic...
print("Abstract Outcome: " + abstract_outcome + " Warnings: " + ", ".join(warnings))  # or fold into your scoring logic
# For now, we just print some dummy results
print(f"ok;50%")
print(f"divide by zero;50%") 
print(f"assertion error;50%")
print(f"out of bounds;50%")
print(f"null pointer;50%")
print(f"*;50%")