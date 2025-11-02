"""
Supporter Agent - Hidden Agent
Agent ẩn hỗ trợ web generation flow
Khi người dùng muốn tạo web, sẽ gọi test_gpt5.py để generate website đàng hoàng
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Default spec from quick_fix_test.py
DEFAULT_WEB_SPEC = {
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


class SupporterAgent:
    """
    Hidden agent that supports web generation by calling test_gpt5.py logic
    """
    
    def __init__(self):
        self.agent_name = "Supporter Agent"
        self.is_hidden = True  # Hidden from UI
        logger.info("SupporterAgent initialized (hidden)")
    
    def quick_fix_web(self, spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call test_gpt5.py logic to generate website đàng hoàng
        
        Args:
            spec: Web specification JSON. If None, uses DEFAULT_WEB_SPEC from quick_fix_test.py
            
        Returns:
            Dictionary with generated code paths and info
        """
        try:
            # Import WebGeneratorAgent (similar to test_gpt5.py)
            from src.agents.web_generator_agent import WebGeneratorAgent
            
            logger.info("SupporterAgent: Calling test_gpt5.py logic for web generation")
            
            # Use default spec from quick_fix_test.py if spec is None or empty
            if not spec or (isinstance(spec, dict) and not spec):
                spec = DEFAULT_WEB_SPEC.copy()
                logger.info("SupporterAgent: Using default spec from quick_fix_test.py")
            
            # Initialize WebGeneratorAgent with GPT-4o model
            agent = WebGeneratorAgent(model="gpt-4o")
            
            # Generate web app with HTML framework (like test_gpt5.py)
            logger.info("SupporterAgent: Starting web app generation with GPT-4o...")
            result = agent.generate_web_app(
                spec=spec,
                framework="html"  # HTML framework for better quality
            )
            
            logger.info("SupporterAgent: Web app generation completed, saving files...")
            
            # Save to output directory
            output_dir = project_root / "fixed_website"
            agent.save_generated_code(result, str(output_dir))
            
            # Verify files were created and have content
            html_path = output_dir / "index.html"
            css_path = output_dir / "styles.css"
            js_path = output_dir / "script.js"
            
            # Wait and verify files exist with content
            import time
            max_wait = 500  # Wait up to 120 seconds (2 minutes) for web generation
            wait_count = 0
            while wait_count < max_wait:
                if html_path.exists() and html_path.stat().st_size > 0:
                    # Read and verify HTML content
                    with open(html_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    # Verify HTML has actual structure (not just text)
                    if len(html_content) > 500 and ('<!DOCTYPE' in html_content or '<html' in html_content):
                        logger.info(f"SupporterAgent: HTML file verified with {len(html_content)} characters")
                        break
                    else:
                        logger.warning(f"SupporterAgent: HTML file exists but content seems incomplete ({len(html_content)} chars)")
                else:
                    logger.info(f"SupporterAgent: Waiting for HTML file to be created... ({wait_count}s)")
                
                time.sleep(1)
                wait_count += 1
            
            if not html_path.exists() or html_path.stat().st_size == 0:
                logger.error("SupporterAgent: HTML file was not created or is empty!")
                raise FileNotFoundError(f"HTML file not found at {html_path}")
            
            logger.info(f"SupporterAgent: Web generated successfully at {output_dir}")
            logger.info(f"SupporterAgent: HTML: {html_path} ({html_path.stat().st_size} bytes)")
            logger.info(f"SupporterAgent: CSS: {css_path} ({css_path.stat().st_size if css_path.exists() else 0} bytes)")
            logger.info(f"SupporterAgent: JS: {js_path} ({js_path.stat().st_size if js_path.exists() else 0} bytes)")
            
            return {
                "success": True,
                "output_dir": str(output_dir),
                "html_path": str(html_path),
                "css_path": str(css_path) if css_path.exists() else None,
                "js_path": str(js_path) if js_path.exists() else None,
                "message": "Website generated successfully using GPT-4o!"
            }
            
        except Exception as e:
            logger.error(f"SupporterAgent error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "message": f"Error generating website: {str(e)}"
            }


# Singleton instance
_supporter_agent_instance: Optional[SupporterAgent] = None


def get_supporter_agent() -> SupporterAgent:
    """Get singleton SupporterAgent instance"""
    global _supporter_agent_instance
    if _supporter_agent_instance is None:
        _supporter_agent_instance = SupporterAgent()
    return _supporter_agent_instance


def quick_fix_web_generation(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public function to call quick fix web generation
    
    Args:
        spec: Web specification JSON
        
    Returns:
        Dictionary with result
    """
    agent = get_supporter_agent()
    return agent.quick_fix_web(spec)

