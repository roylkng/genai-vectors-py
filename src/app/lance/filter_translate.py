"""
AWS S3 Vectors filter to Lance WHERE clause translation.

Translates AWS-style JSON filters to Lance SQL WHERE clauses.
"""

from typing import Dict, Any


def aws_filter_to_where(filter_doc: Dict[str, Any]) -> str:
    """
    Convert AWS S3 Vectors filter to Lance WHERE clause.
    
    Since we only have a nonfilter JSON column, all filters operate on JSON.
    Supports enhanced filtering with new operators for AWS parity.
    
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
    Enhanced to support all AWS S3 Vectors operators.
    
    AWS format examples:
    - {"operator": "equals", "metadata_key": "category", "value": "test"}
    - {"operator": "in", "metadata_key": "status", "value": ["active", "pending"]}
    - {"operator": "and", "conditions": [...]}
    - {"operator": "or", "conditions": [...]}
    """
    op = filter_doc.get("operator")
    # S3-compatible logical operators
    if op in ["and", "$and"]:
        conditions = filter_doc.get("conditions") or filter_doc.get("operands") or filter_doc.get("value")
        if not conditions:
            return "TRUE"
        sql_conditions = [_translate_aws_filter(cond) for cond in conditions]
        return f"({' AND '.join(sql_conditions)})"
    if op in ["or", "$or"]:
        conditions = filter_doc.get("conditions") or filter_doc.get("operands") or filter_doc.get("value")
        if not conditions:
            return "TRUE"
        sql_conditions = [_translate_aws_filter(cond) for cond in conditions]
        return f"({' OR '.join(sql_conditions)})"
    # S3-compatible leaf operators
    metadata_key = filter_doc.get("metadata_key")
    value = filter_doc.get("value")
    if not metadata_key:
        return "TRUE"
    column = f'"{metadata_key}"'
    if op in ["equals", "$eq"]:
        return f"{column} = {format_sql_value(value)}"
    if op in ["not_equals", "$ne"]:
        return f"{column} != {format_sql_value(value)}"
    if op in ["greater_than", "$gt"]:
        return f"{column} > {format_sql_value(value)}"
    if op in ["greater_equal", "$gte"]:
        return f"{column} >= {format_sql_value(value)}"
    if op in ["less_than", "$lt"]:
        return f"{column} < {format_sql_value(value)}"
    if op in ["less_equal", "$lte"]:
        return f"{column} <= {format_sql_value(value)}"
    if op in ["in", "$in"]:
        if not isinstance(value, list) or not value:
            return "FALSE"
        value_list = ', '.join(format_sql_value(v) for v in value)
        return f"{column} IN ({value_list})"
    if op in ["not_in", "$nin"]:
        if not isinstance(value, list) or not value:
            return "TRUE"
        value_list = ', '.join(format_sql_value(v) for v in value)
        return f"{column} NOT IN ({value_list})"
    if op in ["exists", "$exists"]:
        if value:
            return f"{column} IS NOT NULL"
        else:
            return f"{column} IS NULL"
    return "TRUE"

# Helper to format SQL values for correct type
def format_sql_value(val):
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    elif isinstance(val, (int, float)):
        return str(val)
    else:
        return f"'{val}'"
            # String contains operation


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
