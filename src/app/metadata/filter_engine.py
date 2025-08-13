from typing import Dict, Any

def _cmp_num(op: str, val, arg) -> bool:
    if not isinstance(val, (int, float)): return False
    if op == "gt": return val > arg
    if op == "gte": return val >= arg
    if op == "lt": return val < arg
    if op == "lte": return val <= arg
    return False

def matches(md: Dict[str, Any], flt: Dict[str, Dict[str, Any]]) -> bool:
    if not flt: return True
    for k, ops in flt.items():
        has = k in md
        val = md.get(k)
        for op, cond in ops.items():
            if op == "exists":
                if bool(cond) != has: return False
            elif op == "eq":
                if val != cond: return False
            elif op == "neq":
                if val == cond: return False
            elif op in ("gt","gte","lt","lte"):
                if not _cmp_num(op, val, cond): return False
            elif op == "in":
                if val not in cond: return False
            elif op == "nin":
                if val in cond: return False
            else:
                return False
    return True
