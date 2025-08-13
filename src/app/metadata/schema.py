from pydantic import BaseModel
from typing import List, Literal

class FilterableKey(BaseModel):
    name: str
    type: Literal["int64","float64","bool","string","string[]"]

class MetadataSchema(BaseModel):
    filterable: List[FilterableKey] = []
    nonfilterable: List[str] = []
