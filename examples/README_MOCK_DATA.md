# Mock Data Setup

Script này tạo mock data để test RAG Agent.

## Chạy script

```bash
python examples/create_mock_data.py
```

## Yêu cầu

1. **Milvus** phải đang chạy (default: `localhost:19530`)
2. **Neo4j** phải đang chạy (default: `bolt://localhost:7687`)
3. Cần có các packages: `sentence-transformers`, `pymilvus`, `neo4j`

## Dữ liệu được tạo

### Milvus (KB) - 5 Document Chunks:

1. Project Alpha description
2. Requirement REQ-001 (User Authentication)
3. User Story US-001 (Order History)
4. Project Beta description
5. Requirement REQ-002 (Role-Based Access Control)

### Neo4j (KG):

**Projects:**
- PROJ-001: Project Alpha
- PROJ-002: Project Beta

**Requirements:**
- REQ-001: User Authentication (high priority, approved)
- REQ-002: Role-Based Access Control (high priority, approved)
- REQ-003: Order History View (medium priority, draft)

**User Stories:**
- US-001: View Order History (high priority, backlog)
- US-002: Secure Login (critical priority, in_progress)
- US-003: Manage User Roles (high priority, backlog)

**Relationships:**
- Project → CONTAINS_REQUIREMENT → Requirements
- Project → CONTAINS_STORY → User Stories
- Requirement → DERIVED_FROM → User Story
- Requirement → DEPENDS_ON → Requirement
- UserStory → RELATES_TO → User Story

## Lưu ý

- Script sẽ xóa tất cả Project, Requirement, UserStory nodes hiện có trước khi tạo mock data
- Nếu muốn giữ lại data cũ, comment dòng `session.run("MATCH (n) WHERE n:Project OR n:Requirement OR n:UserStory DETACH DELETE n")` trong script
- Milvus sẽ insert mới nếu collection trống, hoặc skip nếu đã có data

## Test sau khi tạo mock data

```bash
python examples/test_rag_simple.py
```

Hoặc interactive mode:

```bash
python examples/test_rag_agent.py --mode interactive
```

## Troubleshooting

1. **Milvus connection error**: Đảm bảo Milvus đang chạy
2. **Neo4j connection error**: Đảm bảo Neo4j đang chạy và đúng credentials
3. **Sparse vector error**: Script dùng empty dicts cho sparse vectors, nếu lỗi có thể cần format khác



