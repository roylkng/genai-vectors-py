"""
AWS S3 Vectors filter to Lance WHERE clause translation.

Translates AWS-style JSON filters to Lance SQL WHERE clauses.
"""

from typing import Dict, Any


def key_expr(table, key: str) -> str:
    """
    Resolve metadata key to appropriate SQL expression.
    
    Prefers typed columns for efficiency; falls back to JSON extraction.
    
    Args:
        table: Lance table object
        key: Metadata key name
        
    Returns:
        SQL expression for accessing the key
    """
    # Get available columns
    cols = set()
    if hasattr(table, "schema") and hasattr(table.schema, "names"):
        cols = set(table.schema.names)
    
    # Prefer typed column if it exists
    if key in cols:
        return f'"{key}"'  # typed column
    
    # Fall back to JSON extraction
    return f"json_extract(metadata_json, '$.{key}')"


def aws_filter_to_where(filter_doc: Dict[str, Any], table=None) -> str:
    """
    Convert AWS S3 Vectors filter to Lance WHERE clause.
    
    Prefers typed columns for efficiency; falls back to JSON extraction.
    
    Args:
        filter_doc: AWS S3 Vectors filter document
        table: Optional Lance table for schema-aware translation
        
    Returns:
        Lance SQL WHERE clause string
    """
    if not filter_doc:
        return ""
    
    return _translate_aws_filter(filter_doc, table)


def _translate_aws_filter(filter_doc: Dict[str, Any], table=None) -> str:
    """
    Translate AWS S3 Vectors filter format to SQL WHERE clause.
    
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
        sql_conditions = [_translate_aws_filter(cond, table) for cond in conditions]
        return f"({' AND '.join(sql_conditions)})"
    if op in ["or", "$or"]:
        conditions = filter_doc.get("conditions") or filter_doc.get("operands") or filter_doc.get("value")
        if not conditions:
            return "TRUE"
        sql_conditions = [_translate_aws_filter(cond, table) for cond in conditions]
        return f"({' OR '.join(sql_conditions)})"
    # S3-compatible leaf operators
    metadata_key = filter_doc.get("metadata_key")
    value = filter_doc.get("value")
    if not metadata_key:
        return "TRUE"
    
    # Use schema-aware key expression
    column = key_expr(table, metadata_key) if table else f"json_extract(metadata_json, '$.{metadata_key}')"
    
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

def format_sql_value(val: Any) -> str:
    """Format a Python value for use in SQL, handling bool, number, and string types."""
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    elif isinstance(val, (int, float)):
        return str(val)
    else:
        # Escape single quotes for SQL safety
        escaped_val = str(val).replace("'", "''")
        return f"'{escaped_val}'"


