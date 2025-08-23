"""
AWS S3 Vectors filter to Lance WHERE clause translation.

Translates AWS-style JSON filters to Lance SQL WHERE clauses.
"""

import json
from typing import Dict, Any


def aws_filter_to_where(filter_doc: Dict[str, Any]) -> str:
    """
    Convert AWS S3 Vectors filter to Lance WHERE clause.
    
    Since we only have a nonfilter JSON column, all filters operate on JSON.
    
    Args:
        filter_doc: AWS S3 Vectors filter document
        
    Returns:
        Lance SQL WHERE clause string
    """
    if not filter_doc:
        return ""
    
    return _translate_aws_filter(filter_doc)


def _translate_aws_filter(filter_doc: Dict[str, Any]) -> str:
    """
    Translate AWS S3 Vectors filter format to SQL WHERE clause.
    
    AWS format examples:
    - {"operator": "equals", "metadata_key": "category", "value": "test"}
    - {"operator": "and", "operands": [...]}
    """
    operator = filter_doc.get("operator")
    
    if operator == "and":
        operands = filter_doc.get("operands", [])
        conditions = [_translate_aws_filter(op) for op in operands]
        return f"({' AND '.join(conditions)})"
    
    elif operator == "or":
        operands = filter_doc.get("operands", [])
        conditions = [_translate_aws_filter(op) for op in operands]
        return f"({' OR '.join(conditions)})"
    
    elif operator == "not":
        operand = filter_doc.get("operand")
        if operand:
            condition = _translate_aws_filter(operand)
            return f"NOT ({condition})"
        return "TRUE"
    
    else:
        # Leaf condition - operate on JSON column
        metadata_key = filter_doc.get("metadata_key")
        value = filter_doc.get("value")
        
        if not metadata_key:
            return "TRUE"
        
        # Use JSON functions to query the nonfilter column
        escaped_key = _escape_json_key(metadata_key)
        escaped_value = _escape_value(value)
        
        if operator == "equals":
            return f"json_extract(nonfilter, '$.{escaped_key}') = {escaped_value}"
        elif operator == "not_equals":
            return f"json_extract(nonfilter, '$.{escaped_key}') != {escaped_value}"
        elif operator == "greater_than":
            return f"CAST(json_extract(nonfilter, '$.{escaped_key}') AS DOUBLE) > {value}"
        elif operator == "greater_than_or_equal":
            return f"CAST(json_extract(nonfilter, '$.{escaped_key}') AS DOUBLE) >= {value}"
        elif operator == "less_than":
            return f"CAST(json_extract(nonfilter, '$.{escaped_key}') AS DOUBLE) < {value}"
        elif operator == "less_than_or_equal":
            return f"CAST(json_extract(nonfilter, '$.{escaped_key}') AS DOUBLE) <= {value}"
        elif operator == "exists":
            if value:
                return f"json_extract(nonfilter, '$.{escaped_key}') IS NOT NULL"
            else:
                return f"json_extract(nonfilter, '$.{escaped_key}') IS NULL"
        else:
            # Unknown operator
            return "TRUE"


def _escape_json_key(key: str) -> str:
    """Escape a JSON key for use in json_extract."""
    return key.replace('"', '\\"')


def _escape_value(value: Any) -> str:
    """Escape a value for SQL."""
    if value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        return f"'{value.replace(chr(39), chr(39) + chr(39))}'"  # Escape single quotes
    else:
        return f"'{str(value).replace(chr(39), chr(39) + chr(39))}'"
