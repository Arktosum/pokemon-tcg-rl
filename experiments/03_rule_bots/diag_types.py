"""Check if dataclass field types are strings or actual types in Python 3.13."""
import sys
sys.path.insert(0, "C:/Users/siddh/miniconda3/Lib/site-packages")
_this = "g:/programming/github-repositories/pokemon-tcg-rl/experiments/01_baseline/agent"
sys.path.insert(0, _this)

from cg.api import Option, SelectData, Observation
from dataclasses import fields
import typing

print("Python:", sys.version)
print()

for cls in [Option, SelectData, Observation]:
    print(f"=== {cls.__name__} ===")
    for f in fields(cls):
        ft = f.type
        print(f"  {f.name}: repr={ft!r}, type={type(ft).__name__}, "
              f"is_str={isinstance(ft, str)}, has_args={hasattr(ft, '__args__')}")
        if hasattr(ft, '__args__'):
            print(f"    __args__={[str(a) for a in ft.__args__]}")
    print()

# Now test to_dataclass with a Struct-like input
from cg.utils import to_dataclass

test_option = {"type": 13, "attackId": 10, "cardId": None, "serial": None,
               "area": None, "index": None, "playerIndex": None,
               "toolIndex": None, "energyIndex": None, "count": None,
               "inPlayArea": None, "inPlayIndex": None, "cardId": None,
               "specialConditionType": None}
try:
    result = to_dataclass(test_option, Option)
    print(f"Option result type: {type(result)}")
    print(f"  attackId: {result.attackId}")
    print(f"  has attackId attr: {hasattr(result, 'attackId')}")
except Exception as e:
    print(f"to_dataclass ERROR: {e}")
    import traceback
    traceback.print_exc()
