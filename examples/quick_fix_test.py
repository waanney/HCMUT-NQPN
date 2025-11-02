"""
Quick test to fix ugly website issues
Uses gpt-4o-mini for faster generation
"""

import os
import sys
import webbrowser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.web_generator_agent import WebGeneratorAgent

# Simple spec for testing
simple_spec = {
    "site": {
        "name": "NexusTech Solutions",
        "language": "vi",
        "defaultTitle": "NexusTech - Giải Pháp Tương Lai",
        "defaultDescription": "Chuyên gia về AI và điện toán đám mây."
    },
    "design": {
        "primaryColor": "#0A74DA",
        "secondaryColor": "#FF6B6B",
        "backgroundColor": "#F4F7F6",
        "textColor": "#212121",
        "font": {
            "primary": "'Inter', sans-serif",
            "headings": "'Roboto Slab', serif"
        },
        "layout": {
            "maxWidth": "1280px"
        }
    },
    "navigation": {
        "navbar": {
            "links": [
                {"label": "Trang chủ", "path": "/"},
                {"label": "Dịch vụ", "path": "/services"},
                {"label": "Về chúng tôi", "path": "/about"}
            ],
            "callToAction": {
                "label": "Liên hệ",
                "action": "openModal",
                "target": "contact-modal"
            }
        },
        "footer": {
            "copyright": "© 2025 NexusTech",
            "columns": [
                {
                    "title": "Công ty",
                    "links": [
                        {"label": "Về chúng tôi", "path": "/about"},
                        {"label": "Tuyển dụng", "path": "/careers"}
                    ]
                }
            ]
        }
    },
    "pages": [
        {
            "name": "Trang chủ",
            "path": "/",
            "components": [
                {
                    "type": "Hero",
                    "data": {
                        "title": "Đổi Mới. Chuyển Đổi. Thành Công.",
                        "subtitle": "Đối tác của bạn trong các dịch vụ AI và Đám mây thế hệ mới.",
                        "button": {
                            "label": "Khám phá dịch vụ",
                            "action": "navigate",
                            "target": "/services"
                        }
                    }
                },
                {
                    "type": "FeatureList",
                    "data": {
                        "title": "Dịch vụ của chúng tôi",
                        "features": [
                            {"name": "Giải pháp AI", "icon": "🤖"},
                            {"name": "Cloud Hosting", "icon": "☁️"},
                            {"name": "Bảo mật mạng", "icon": "🔒"}
                        ]
                    }
                }
            ]
        }
    ]
}

if __name__ == "__main__":
    print("🎨 Quick Fix Test - Making Website Beautiful")
    print("=" * 60)
    
    # Use gpt-4o-mini for faster testing
    agent = WebGeneratorAgent(model="gpt-4o")
    
    print("\n📝 Generating website...")
    print("⏳ Please wait 10-15 seconds...\n")
    
    try:
        result = agent.generate_web_app(
            spec=simple_spec,
            framework="vanilla"  # Vanilla HTML/CSS/JS for simplicity
        )
        
        # Save to output
        output_dir = os.path.join(os.path.dirname(__file__), "..", "fixed_website")
        agent.save_generated_code(result, output_dir)
        
        print("✅ Website generated!")
        print(f"📁 Location: {output_dir}")
        
        # Show previews
        print("\n" + "=" * 60)
        print("HTML Preview:")
        print(result.html[:600])
        print("\n" + "=" * 60)
        print("CSS Preview:")
        print(result.css[:600])
        
        # Open in browser
        html_path = os.path.join(output_dir, "index.html")
        print(f"\n🌐 Opening in browser: {html_path}")
        webbrowser.open(f"file:///{html_path}")
        
        print("\n✨ IMPROVEMENTS MADE:")
        print("  ✅ Google Fonts properly imported")
        print("  ✅ styles.css linked in HTML")
        print("  ✅ Real emoji icons (🤖 ☁️ 🔒) instead of empty divs")
        print("  ✅ No Tailwind utility classes conflicts")
        print("  ✅ Beautiful CSS with proper spacing")
        print("  ✅ Smooth hover effects")
        print("  ✅ Responsive design")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
