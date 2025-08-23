# AWS S3 Vectors Parity Implementation Plan

## Current Status Assessment

### ✅ Already Implemented
- Basic resource model (buckets, indexes, vectors)
- Lance storage backend with S3 integration
- Core CRUD operations
- Basic filtering (equality only)
- OpenAPI documentation
- Docker deployment

### ⚠️ Partially Implemented
- Request/response models (missing some AWS-specific fields)
- Error handling (basic HTTP errors, missing AWS error codes)
- ARN support (format exists but not fully utilized)
- Pagination (basic implementation, missing AWS paginator tokens)

### ❌ Missing for Full Parity

#### 1. Request/Response Parity
- [ ] Exact boto3 parameter names and response structures
- [ ] AWS error codes (ValidationException, ResourceNotFoundException, etc.)
- [ ] Proper ARN handling and resolution
- [ ] Missing pagination tokens (NextToken/MaxResults)

#### 2. GetVectors API
- [ ] Fetch vectors by keys (currently missing)
- [ ] Proper returnData/returnMetadata handling
- [ ] Batch key lookup optimization

#### 3. Enhanced Filter Model
- [ ] Support for operators: equals, in, numeric comparisons
- [ ] AND/OR logical operators
- [ ] Filterable vs non-filterable metadata semantics
- [ ] Size/key/type limits validation

#### 4. Index Maintenance Strategy
- [ ] Background optimize() on thresholds
- [ ] fastSearch flag for indexed-only queries
- [ ] Incremental index updates instead of full rebuild

#### 5. Limits/Validation
- [ ] Dimension limits (1-4096)
- [ ] Vector count quotas
- [ ] Metadata size/key limits
- [ ] Batch size validation
- [ ] ValidationException responses

#### 6. Security & Auth
- [ ] SigV4 authentication (or pass-through)
- [ ] Permission-based error mapping
- [ ] IAM-style policy validation

#### 7. Permission Coupling
- [ ] returnMetadata requires GetVectors permission
- [ ] Filter usage requires QueryVectors + GetVectors
- [ ] Proper 403 responses

## Implementation Priority

### Phase 1: Core Parity (High Impact)
1. **GetVectors API** - Critical for client hydration patterns
2. **Enhanced Filters** - operators, AND/OR, validation
3. **Error Code Alignment** - AWS-compatible error responses
4. **Pagination Tokens** - Proper NextToken implementation

### Phase 2: Performance & Scale (Medium Impact)
5. **Index Maintenance** - Background optimization
6. **Limits Validation** - AWS quota enforcement
7. **Request/Response Polish** - Exact field matching

### Phase 3: Production Hardening (Lower Impact)
8. **Authentication** - SigV4 or token-based auth
9. **Permission Coupling** - IAM-style authorization
10. **ARN Resolution** - Full ARN parsing and routing

## Detailed Implementation Tasks

### Task 1: GetVectors API Implementation

**Files to modify:**
- `src/app/models.py` - Add GetVectors request/response models
- `src/app/api.py` - Implement get_vectors endpoint
- `src/app/lance/index_ops.py` - Add batch key lookup

**Expected outcome:**
```python
# New endpoint
@router.post("/buckets/{bucket_name}/indexes/{index_name}/vectors:get")
async def get_vectors(bucket_name: str, index_name: str, request: GetVectorsRequest)
```

### Task 2: Enhanced Filter Model

**Files to modify:**
- `src/app/models.py` - Expand MetadataFilter model
- `src/app/lance/filter_translate.py` - Support new operators
- `src/app/metadata/filter_engine.py` - Add validation logic

**Filter operators to support:**
- `equals`, `not_equals`
- `in`, `not_in`
- `greater_than`, `less_than`, `greater_equal`, `less_equal`
- `and`, `or` logical operators

### Task 3: AWS Error Code Alignment

**Files to modify:**
- `src/app/api.py` - Replace HTTPException with AWS error codes
- Create `src/app/errors.py` - AWS error response models

**Error codes to implement:**
- `ValidationException` - Invalid parameters
- `ResourceNotFoundException` - Missing bucket/index
- `ConflictException` - Resource already exists
- `AccessDeniedException` - Permission denied
- `ThrottlingException` - Rate limiting

### Task 4: Pagination Implementation

**Files to modify:**
- `src/app/models.py` - Add proper NextToken fields
- `src/app/lance/index_ops.py` - Implement cursor-based pagination
- `src/app/api.py` - Use pagination in list operations

**Implementation strategy:**
- Use Lance's built-in pagination with cursors
- Base64-encode cursor tokens for NextToken
- Implement MaxResults limiting

### Task 5: Index Maintenance Strategy

**Files to modify:**
- `src/app/lance/index_ops.py` - Background optimization
- `src/app/util/config.py` - Optimization thresholds
- Create `src/app/index/manager.py` - Background task management

**Strategy:**
- Track row counts and time since last optimize
- Background optimize() based on configurable thresholds
- Add fastSearch parameter to restrict to indexed rows

### Task 6: Limits and Validation

**Files to modify:**
- `src/app/models.py` - Add validation constraints
- Create `src/app/validation/` - AWS limits enforcement
- `src/app/api.py` - Pre-flight validation

**Limits to enforce:**
- Vector dimensions: 1-4096
- Metadata size: 8KB per vector
- Metadata keys: 50 per vector
- Batch size: 100 vectors per request
- Index count: 20 per bucket

## Testing Strategy

### Unit Tests
- Test each new API endpoint
- Validate filter translation
- Check error code mapping
- Verify pagination logic

### Integration Tests
- Test with real boto3 client
- Verify AWS CLI compatibility
- Performance benchmarks
- Error handling scenarios

### Compatibility Tests
- Compare responses with real AWS S3 Vectors
- Validate exact field matching
- Test pagination across large datasets
- Verify filter behavior parity

## Success Criteria

### Phase 1 Complete When:
- [ ] boto3 client can perform all CRUD operations without modification
- [ ] GetVectors API returns vectors by keys with proper metadata
- [ ] Advanced filters (in, numeric, AND/OR) work correctly
- [ ] All error responses match AWS error codes
- [ ] Pagination works with real NextToken values

### Phase 2 Complete When:
- [ ] Index optimization happens in background
- [ ] All AWS limits are enforced with proper ValidationException
- [ ] Performance matches or exceeds current implementation
- [ ] Large datasets can be paginated efficiently

### Phase 3 Complete When:
- [ ] Authentication system prevents unauthorized access
- [ ] Permission coupling enforces AWS-style authorization
- [ ] ARN resolution allows targeting by ARN or name
- [ ] Full production deployment ready

## Implementation Timeline

**Week 1-2:** GetVectors API + Enhanced Filters
**Week 3-4:** Error Codes + Pagination  
**Week 5-6:** Index Maintenance + Limits
**Week 7-8:** Authentication + Polish

This plan will bring the implementation to full AWS S3 Vectors parity while maintaining the current Lance-based performance advantages.
