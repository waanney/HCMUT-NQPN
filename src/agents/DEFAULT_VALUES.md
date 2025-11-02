# 📋 Default Values - Simple Suggestion Agent

## Tổng quan

Simple Suggestion Agent có sẵn **default values** cho các trường thường gặp. Khi phát hiện data thiếu, agent sẽ:

1. **Interactive Mode**: Hỏi user chọn giữa default hoặc custom
2. **Auto Mode**: Tự động dùng default values

## 🎯 Các trường có Default Values

### 1. Site Information

| Field Path | Default Value | Mô tả |
|------------|---------------|-------|
| `site.name` | `"My Website"` | Tên website |
| `site.language` | `"vi"` | Ngôn ngữ (Vietnamese) |
| `site.defaultTitle` | `"Welcome to My Website"` | Title mặc định |
| `site.defaultDescription` | `"A professional website built with modern technology"` | Description mặc định |

### 2. Design System - Colors

| Field Path | Default Value | Mô tả |
|------------|---------------|-------|
| `design.primaryColor` | `"#0066CC"` | Màu chính (xanh dương) |
| `design.secondaryColor` | `"#FF6B6B"` | Màu phụ (đỏ cam) |
| `design.backgroundColor` | `"#FFFFFF"` | Màu nền (trắng) |
| `design.textColor` | `"#333333"` | Màu chữ (xám đậm) |

### 3. Design System - Typography

| Field Path | Default Value | Mô tả |
|------------|---------------|-------|
| `design.font.primary` | `"'Inter', sans-serif"` | Font chính |
| `design.font.headings` | `"'Poppins', sans-serif"` | Font headings |

### 4. Design System - Layout

| Field Path | Default Value | Mô tả |
|------------|---------------|-------|
| `design.layout.maxWidth` | `"1200px"` | Chiều rộng tối đa |
| `design.layout.roundedCorners` | `"medium"` | Bo góc (8-12px) |

### 5. Navigation - Navbar

| Field Path | Default Value | Mô tả |
|------------|---------------|-------|
| `navigation.navbar.position` | `"sticky-top"` | Vị trí navbar |
| `navigation.navbar.links` | Array (xem bên dưới) | Navigation links |

**Default Links:**
```json
[
  {"label": "Home", "path": "/"},
  {"label": "About", "path": "/about"},
  {"label": "Contact", "path": "/contact"}
]
```

### 6. Navigation - Footer

| Field Path | Default Value | Mô tả |
|------------|---------------|-------|
| `navigation.footer.copyright` | `"© 2025 All Rights Reserved"` | Copyright text |
| `navigation.footer.columns` | Array (xem bên dưới) | Footer columns |

**Default Columns:**
```json
[
  {
    "title": "Company",
    "links": [
      {"label": "About Us", "path": "/about"},
      {"label": "Careers", "path": "/careers"}
    ]
  },
  {
    "title": "Legal",
    "links": [
      {"label": "Privacy", "path": "/privacy"},
      {"label": "Terms", "path": "/terms"}
    ]
  }
]
```

## 🚫 Các trường KHÔNG có Default

Những trường sau **KHÔNG có default** và cần user input:

### Pages & Components

- `pages` - Cần user định nghĩa cấu trúc pages
- `pages[*].name` - Tên page
- `pages[*].path` - Route path
- `pages[*].components` - Array components của page
- `pages[*].components[*].type` - Loại component
- `pages[*].components[*].data` - Data của component

**Lý do:** Pages và components phụ thuộc vào business logic và requirements cụ thể của từng project.

## 💡 Interactive Mode Flow

Khi chạy ở Interactive Mode:

```
Field: site.name
   Default: My Website
   
   Lựa chọn:
   [1] Dùng giá trị default
   [2] Nhập giá trị custom
   [3] Bỏ qua (để trống)
   
   Chọn (1/2/3): _
```

### Option 1: Dùng Default
```
Chọn (1/2/3): 1
✅ Đã chọn default
```

### Option 2: Custom Input
```
Chọn (1/2/3): 2
Nhập giá trị: NexusTech Solutions
✅ Đã nhập custom: NexusTech Solutions
```

### Option 3: Bỏ qua
```
Chọn (1/2/3): 3
⏭️ Bỏ qua field này
```

## 🤖 Auto Mode

Khi chạy ở Auto Mode (không hỏi user):

```python
suggestion_agent.suggest_missing_data(missing_fields, interactive=False)
```

Output:
```
💡 Simple Suggestion Agent đang phân tích...
   Số field cần suggest: 8
   ✓ site.name → My Website
   ✓ site.language → vi
   ✓ design.primaryColor → #0066CC
   ✓ design.font.primary → 'Inter', sans-serif
   ✓ navigation.navbar.links → default value
   ...

💡 Đã tạo suggestions cho 8 fields
```

## 📝 Cách sử dụng

### Auto Mode (Không hỏi user)

```python
from src.agents.simple_suggestion_agent import SimpleSuggestionAgent

agent = SimpleSuggestionAgent()

missing_fields = [
    "site.name",
    "design.primaryColor",
    "navigation.navbar.links"
]

# Auto mode - dùng defaults
suggestions = agent.suggest_missing_data(missing_fields, interactive=False)
```

### Interactive Mode (Hỏi user)

```python
from src.agents.simple_suggestion_agent import SimpleSuggestionAgent

agent = SimpleSuggestionAgent()

missing_fields = [
    "site.name",
    "design.primaryColor",
    "navigation.navbar.links"
]

# Interactive mode - hỏi user chọn
suggestions = agent.suggest_missing_data(missing_fields, interactive=True)
```

## 🔧 Thêm Default Values mới

Để thêm default cho field mới, edit `simple_suggestion_agent.py`:

```python
self.default_suggestions = {
    # ... existing defaults ...
    
    # Thêm default mới
    "design.animation.duration": "300ms",
    "design.spacing.unit": "4px",
    # ...
}
```

Hoặc thêm vào `_get_default_value()` cho complex types:

```python
def _get_default_value(self, field_path: str) -> Any:
    # ... existing code ...
    
    # Thêm logic cho field mới
    if "animation" in field_path:
        return {
            "duration": "300ms",
            "easing": "ease-in-out"
        }
```

## 📊 Thống kê Default Values

### Tổng số defaults có sẵn

- **Simple values**: 13 fields
- **Array values**: 2 fields (links, columns)
- **Total**: 15 fields có default

### Phân loại theo section

- **Site**: 4 defaults
- **Design - Colors**: 4 defaults
- **Design - Typography**: 2 defaults
- **Design - Layout**: 2 defaults
- **Navigation - Navbar**: 2 defaults (position + links)
- **Navigation - Footer**: 2 defaults (copyright + columns)

## ✅ Best Practices

### 1. Khi nào dùng Auto Mode?
- Development/testing nhanh
- CI/CD pipelines
- Default prototypes
- Non-critical projects

### 2. Khi nào dùng Interactive Mode?
- Production projects
- Client projects cần customization
- Brand-specific requirements
- Khi cần user input cho branding

### 3. Customize Defaults
Để phù hợp với brand:
```python
agent = SimpleSuggestionAgent()

# Override defaults
agent.default_suggestions["design.primaryColor"] = "#FF0000"  # Brand color
agent.default_suggestions["site.name"] = "YourCompany"
```

## 🎨 Example Output

### Auto Mode Output
```json
{
  "site": {
    "name": "My Website",
    "language": "vi",
    "defaultTitle": "Welcome to My Website",
    "defaultDescription": "A professional website built with modern technology"
  },
  "design": {
    "primaryColor": "#0066CC",
    "secondaryColor": "#FF6B6B",
    "backgroundColor": "#FFFFFF",
    "textColor": "#333333",
    "font": {
      "primary": "'Inter', sans-serif",
      "headings": "'Poppins', sans-serif"
    },
    "layout": {
      "maxWidth": "1200px",
      "roundedCorners": "medium"
    }
  },
  "navigation": {
    "navbar": {
      "position": "sticky-top",
      "links": [
        {"label": "Home", "path": "/"},
        {"label": "About", "path": "/about"},
        {"label": "Contact", "path": "/contact"}
      ]
    },
    "footer": {
      "copyright": "© 2025 All Rights Reserved",
      "columns": [
        {
          "title": "Company",
          "links": [
            {"label": "About Us", "path": "/about"},
            {"label": "Careers", "path": "/careers"}
          ]
        },
        {
          "title": "Legal",
          "links": [
            {"label": "Privacy", "path": "/privacy"},
            {"label": "Terms", "path": "/terms"}
          ]
        }
      ]
    }
  }
}
```

## 🔗 Integration với Prompt Generator

```python
# Prompt Generator tự động dùng Suggestion Agent
agent = PromptGeneratorAgent(suggestion_agent=SimpleSuggestionAgent())

# Auto mode (mặc định)
json_spec = agent.generate_json(ba_output, auto_request_suggestions=True)

# Interactive mode (cần thêm parameter)
# json_spec = agent.generate_json(ba_output, auto_request_suggestions=True, interactive=True)
```

---

**Tổng kết**: Agent có **15 default values** sẵn sàng cho các trường cơ bản. Pages và components cần user define vì phụ thuộc vào business logic cụ thể.
