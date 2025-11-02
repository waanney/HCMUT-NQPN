"""
Test GPT-5 with Web Generator Agent
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.web_generator_agent import WebGeneratorAgent
import webbrowser

print("🚀 Testing GPT-5 Model")
print("=" * 60)

# Simple specification
spec = {
    "site": {
        "name": "TechVision AI",
        "language": "vi",
        "description": "Công ty AI hàng đầu Việt Nam"
    },
    "design": {
        "colors": {
            "primary": "#6366f1",
            "secondary": "#ec4899",
            "background": "#f9fafb",
            "text": "#111827"
        },
        "typography": {
            "fontFamily": {
                "primary": "'Inter', sans-serif",
                "headings": "'Poppins', sans-serif"
            }
        },
        "layout": {
            "maxWidth": "1200px"
        }
    }
}

print("\n📝 Testing with GPT-5...")
print("⏳ Generating website with gpt-5...\n")

try:
    # Initialize agent with GPT-5
    agent = WebGeneratorAgent(model="gpt-5")
    
    # Generate website
    result = agent.generate_web_app(spec, framework="html")
    
    print("✅ Generation successful!")
    print(f"📁 Output: {result['output_dir']}")
    
    # Check for unwanted text
    with open(result['files']['html'], 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    with open(result['files']['css'], 'r', encoding='utf-8') as f:
        css_content = f.read()
    
    print("\n" + "=" * 60)
    print("🔍 QUALITY CHECKS:")
    print("=" * 60)
    
    # Check HTML
    html_clean = True
    if not html_content.startswith('<!DOCTYPE'):
        print("❌ HTML: Có text trước <!DOCTYPE")
        html_clean = False
        print(f"   First 100 chars: {html_content[:100]}")
    else:
        print("✅ HTML: Bắt đầu đúng với <!DOCTYPE")
    
    if "Here's" in html_content or "Here is" in html_content:
        print("❌ HTML: Có text giải thích ('Here is/Here's')")
        html_clean = False
    else:
        print("✅ HTML: Không có text giải thích")
    
    if "```" in html_content:
        print("❌ HTML: Có markdown code blocks")
        html_clean = False
    else:
        print("✅ HTML: Không có markdown")
    
    # Check CSS
    css_clean = True
    if css_content.startswith("Here") or css_content.startswith("This"):
        print("❌ CSS: Có text giải thích ở đầu")
        css_clean = False
        print(f"   First 100 chars: {css_content[:100]}")
    else:
        print("✅ CSS: Bắt đầu với CSS code")
    
    if "```" in css_content:
        print("❌ CSS: Có markdown code blocks")
        css_clean = False
    else:
        print("✅ CSS: Không có markdown")
    
    print("\n" + "=" * 60)
    if html_clean and css_clean:
        print("🎉 PERFECT! Code hoàn toàn sạch, không có text thừa!")
    else:
        print("⚠️  Vẫn còn một số vấn đề với output format")
    print("=" * 60)
    
    # Preview
    print(f"\n📄 HTML Preview (first 500 chars):")
    print("-" * 60)
    print(html_content[:500])
    print("...")
    
    print(f"\n🎨 CSS Preview (first 500 chars):")
    print("-" * 60)
    print(css_content[:500])
    print("...")
    
    # Open in browser
    html_path = result['files']['html']
    print(f"\n🌐 Opening in browser: {html_path}")
    webbrowser.open(f"file://{html_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
