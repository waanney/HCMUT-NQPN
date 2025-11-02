# Formatter Agent với Chainlit UI

## Tổng quan

Formatter Agent nhận output từ RAG Agent và format text với color tags:
- `<g></g>` (xanh) - nội dung được thêm
- `<y></y>` (vàng) - nội dung được modify
- `<r></r>` (đỏ) - nội dung được xóa

## Cấu trúc

### Files

1. **`src/agents/formatter_agent.py`**
   - `FormatterAgent` class: xử lý formatting với LLM
   - `FormatterInput` và `FormatterOutput` Pydantic models
   - `create_formatter_agent()`: tạo OpenAI Agent với tools

2. **`src/tools/formatter_tool.py`**
   - `format_text_with_tags()`: function tool để format text với color tags

3. **`app.py`**
   - Chainlit UI app
   - Tích hợp RAG Agent và Formatter Agent
   - Hiển thị formatted text với colors (HTML)

## Cài đặt

1. Cài đặt dependencies:
```bash
pip install chainlit python-dotenv
```

Hoặc nếu dùng pyproject.toml:
```bash
pip install -e .
```

## Chạy Chainlit UI

1. Đảm bảo có file `.env` với `OPENAI_API_KEY`:
```env
OPENAI_API_KEY=your_api_key_here
```

2. **Quan trọng**: Đảm bảo file `.chainlit/config.toml` có setting:
```toml
[features]
unsafe_allow_html = true
```

Setting này cho phép Chainlit render HTML trong messages để hiển thị formatted text với colors. File config đã được cập nhật tự động.

3. Chạy Chainlit app:

Với port mặc định (8000):
```bash
chainlit run app.py
```

Với port tùy chỉnh (7777):
```bash
chainlit run app.py --port 7777
```

Hoặc với uv:
```bash
uv run chainlit run app.py --port 7777
```

Với auto-reload khi code thay đổi:
```bash
chainlit run app.py --port 7777 -w
```

Hoặc:
```bash
uv run chainlit run app.py --port 7777 -w
```

3. Mở browser tại URL được hiển thị:
   - Mặc định: `http://localhost:8000`
   - Với --port 7777: `http://localhost:7777`

## Sử dụng

1. Nhập câu hỏi vào UI
2. RAG Agent sẽ search Knowledge Base (Milvus) và Knowledge Graph (Neo4j)
3. Formatter Agent sẽ format answer với color tags
4. UI sẽ hiển thị formatted text với colors:
   - Green background: content to be added
   - Yellow background: content to be modified
   - Red background: content to be deleted

## Example

Input query: "What features does Project Alpha include?"

Output:
- RAG Agent retrieves information from KB and KG
- Formatter Agent formats answer with tags like:
  - `<g>New feature: Authentication system</g>`
  - `<y>Updated: User dashboard interface</y>`
  - `<r>Removed: Legacy login page</r>`
- UI displays with colored backgrounds

## API Usage

Nếu không dùng Chainlit UI, có thể sử dụng trực tiếp:

```python
from agents.rag_agent import run_rag_query
from agents.formatter_agent import FormatterAgent

# Run RAG query
rag_output = run_rag_query("What features does Project Alpha include?")

# Format with Formatter Agent
formatter = FormatterAgent()
formatter_output = formatter.format_rag_output(rag_output)

print(formatter_output.formatted_text)  # Text with <g>, <y>, <r> tags
print(formatter_output.KB)  # KB references
print(formatter_output.KG)  # KG references
```

## Color Tags

- `<g>content</g>` - Green (background-color: #90EE90) - ADD
- `<y>content</y>` - Yellow (background-color: #FFE4B5) - MODIFY
- `<r>content</r>` - Red (background-color: #FFB6C1) - DELETE

HTML rendering trong Chainlit sẽ tự động parse các tags này thành colored spans.
