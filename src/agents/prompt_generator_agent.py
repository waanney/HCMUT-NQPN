"""
Prompt Generator Agent
Nhiệm vụ: Nhận JSON structure từ BA Agent và tạo prompt hoàn chỉnh cho Rocket code generation
"""

import json
from typing import Dict, Any, List, Optional, Tuple


class PromptGeneratorAgent:
    """
    Agent chuyển đổi Business Analysis JSON thành prompt chi tiết cho Rocket
    Validate data và yêu cầu Suggestion Agent bổ sung data thiếu
    """
    
    # Define required fields cho validation
    REQUIRED_FIELDS = {
        "site": ["name", "language", "defaultTitle", "defaultDescription"],
        "design": ["primaryColor", "secondaryColor", "backgroundColor", "textColor"],
        "design.font": ["primary", "headings"],
        "design.layout": ["maxWidth", "roundedCorners"],
        "navigation.navbar": ["position", "links"],
        "navigation.footer": ["copyright", "columns"],
        "pages": []  # Pages phải có ít nhất 1 page
    }
    
    def __init__(self, suggestion_agent=None):
        self.agent_name = "Prompt Generator Agent"
        self.agent_icon = "🚀"
        self.suggestion_agent = suggestion_agent  # Reference đến Suggestion Agent
        self.missing_fields = []  # Track các field còn thiếu
    
    def validate_ba_output(self, ba_output: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate BA output và tìm các field còn thiếu
        
        Args:
            ba_output: JSON output từ BA Agent
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list of missing fields)
        """
        missing = []
        
        # Validate site fields
        site = ba_output.get("site", {})
        for field in self.REQUIRED_FIELDS["site"]:
            if not site.get(field) or site.get(field) == "":
                missing.append(f"site.{field}")
        
        # Validate design fields
        design = ba_output.get("design", {})
        for field in self.REQUIRED_FIELDS["design"]:
            if not design.get(field) or design.get(field) == "":
                missing.append(f"design.{field}")
        
        # Validate design.font
        font = design.get("font", {})
        for field in self.REQUIRED_FIELDS["design.font"]:
            if not font.get(field) or font.get(field) == "":
                missing.append(f"design.font.{field}")
        
        # Validate design.layout
        layout = design.get("layout", {})
        for field in self.REQUIRED_FIELDS["design.layout"]:
            if not layout.get(field) or layout.get(field) == "":
                missing.append(f"design.layout.{field}")
        
        # Validate navigation
        navigation = ba_output.get("navigation", {})
        navbar = navigation.get("navbar", {})
        for field in self.REQUIRED_FIELDS["navigation.navbar"]:
            if not navbar.get(field) or (isinstance(navbar.get(field), list) and len(navbar.get(field)) == 0):
                missing.append(f"navigation.navbar.{field}")
        
        footer = navigation.get("footer", {})
        for field in self.REQUIRED_FIELDS["navigation.footer"]:
            if not footer.get(field) or (isinstance(footer.get(field), list) and len(footer.get(field)) == 0):
                missing.append(f"navigation.footer.{field}")
        
        # Validate pages
        pages = ba_output.get("pages", [])
        if not pages or len(pages) == 0:
            missing.append("pages (ít nhất 1 page)")
        else:
            # Validate mỗi page
            for idx, page in enumerate(pages):
                if not page.get("name"):
                    missing.append(f"pages[{idx}].name")
                if not page.get("path"):
                    missing.append(f"pages[{idx}].path")
                if not page.get("components") or len(page.get("components", [])) == 0:
                    missing.append(f"pages[{idx}].components (ít nhất 1 component)")
                else:
                    # Validate components
                    for comp_idx, comp in enumerate(page.get("components", [])):
                        if not comp.get("type"):
                            missing.append(f"pages[{idx}].components[{comp_idx}].type")
                        if not comp.get("data"):
                            missing.append(f"pages[{idx}].components[{comp_idx}].data")
        
        is_valid = len(missing) == 0
        return is_valid, missing
    
    def request_missing_data(self, missing_fields: List[str], interactive: bool = True) -> Optional[Dict[str, Any]]:
        """
        Gọi Suggestion Agent để hỏi về data còn thiếu
        HỎI USER lựa chọn giữa default values hoặc custom input
        
        Args:
            missing_fields: List các field còn thiếu
            interactive: True = hỏi user, False = auto dùng default
            
        Returns:
            Dict with suggested data hoặc None nếu không có Suggestion Agent
        """
        if not self.suggestion_agent:
            print(f"⚠️ Không có Suggestion Agent để hỏi về data thiếu!")
            print(f"❌ Các field còn thiếu: {', '.join(missing_fields)}")
            return None
        
        print(f"\n🤔 Phát hiện {len(missing_fields)} field thiếu data:")
        for field in missing_fields:
            print(f"   - {field}")
        
        if interactive:
            print(f"\n💡 Đang gọi Suggestion Agent để HỎI USER...")
        else:
            print(f"\n💡 Đang gọi Suggestion Agent để dùng DEFAULT VALUES...")
        
        # Call Suggestion Agent với interactive mode
        try:
            suggested_data = self.suggestion_agent.suggest_missing_data(
                missing_fields, 
                interactive=interactive
            )
            print(f"✅ Suggestion Agent đã trả về suggestions!")
            return suggested_data
        except Exception as e:
            print(f"❌ Lỗi khi gọi Suggestion Agent: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_suggested_data(
        self, 
        original_data: Dict[str, Any], 
        suggested_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge suggested data vào original data
        
        Args:
            original_data: BA output gốc
            suggested_data: Data từ Suggestion Agent
            
        Returns:
            Merged data
        """
        import copy
        merged = copy.deepcopy(original_data)
        
        # Deep merge dictionaries
        def deep_merge(base, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(merged, suggested_data)
        return merged
    
    def _build_json_specification(self, ba_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xây dựng JSON specification đầy đủ từ validated BA output
        
        Args:
            ba_output: Validated BA output
            
        Returns:
            Dict: Complete JSON specification for Rocket
        """
        # Parse các thành phần
        site_info = ba_output.get("site", {})
        design_config = ba_output.get("design", {})
        navigation = ba_output.get("navigation", {})
        pages = ba_output.get("pages", [])
        
        # Build comprehensive JSON structure
        json_spec = {
            "projectInfo": {
                "name": site_info.get("name", ""),
                "description": "Complete website specification for code generation",
                "version": "1.0.0",
                "generatedBy": "Prompt Generator Agent",
                "targetFramework": "React 18+ with TypeScript",
                "buildTool": "Vite",
                "styling": "Tailwind CSS"
            },
            "siteConfiguration": {
                "name": site_info.get("name", ""),
                "language": site_info.get("language", "vi"),
                "defaultTitle": site_info.get("defaultTitle", ""),
                "defaultDescription": site_info.get("defaultDescription", ""),
                "seo": {
                    "enableMetaTags": True,
                    "enableOpenGraph": True,
                    "enableTwitterCards": True
                }
            },
            "designSystem": {
                "colors": {
                    "primary": design_config.get("primaryColor", ""),
                    "secondary": design_config.get("secondaryColor", ""),
                    "background": design_config.get("backgroundColor", ""),
                    "text": design_config.get("textColor", ""),
                    "success": "#10B981",
                    "warning": "#F59E0B",
                    "error": "#EF4444",
                    "info": "#3B82F6"
                },
                "typography": {
                    "fontFamily": {
                        "primary": design_config.get("font", {}).get("primary", ""),
                        "headings": design_config.get("font", {}).get("headings", ""),
                        "monospace": "'Fira Code', monospace"
                    },
                    "fontSize": {
                        "xs": "0.75rem",
                        "sm": "0.875rem",
                        "base": "1rem",
                        "lg": "1.125rem",
                        "xl": "1.25rem",
                        "2xl": "1.5rem",
                        "3xl": "1.875rem",
                        "4xl": "2.25rem"
                    },
                    "fontWeight": {
                        "light": 300,
                        "normal": 400,
                        "medium": 500,
                        "semibold": 600,
                        "bold": 700
                    }
                },
                "spacing": {
                    "unit": "4px",
                    "scale": [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96]
                },
                "layout": {
                    "maxWidth": design_config.get("layout", {}).get("maxWidth", ""),
                    "borderRadius": design_config.get("layout", {}).get("roundedCorners", ""),
                    "borderRadiusValues": {
                        "small": "4px",
                        "medium": "8px",
                        "large": "12px",
                        "xl": "16px",
                        "full": "9999px"
                    }
                },
                "shadows": {
                    "sm": "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
                    "base": "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
                    "md": "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                    "lg": "0 10px 15px -3px rgba(0, 0, 0, 0.1)",
                    "xl": "0 20px 25px -5px rgba(0, 0, 0, 0.1)"
                },
                "animations": {
                    "duration": {
                        "fast": "150ms",
                        "base": "300ms",
                        "slow": "500ms"
                    },
                    "easing": {
                        "easeIn": "cubic-bezier(0.4, 0, 1, 1)",
                        "easeOut": "cubic-bezier(0, 0, 0.2, 1)",
                        "easeInOut": "cubic-bezier(0.4, 0, 0.2, 1)"
                    }
                }
            },
            "navigation": self._build_navigation_spec(navigation),
            "pages": self._build_pages_spec(pages),
            "technicalRequirements": {
                "frontend": {
                    "framework": "React 18+",
                    "language": "TypeScript",
                    "buildTool": "Vite",
                    "packageManager": "npm"
                },
                "styling": {
                    "framework": "Tailwind CSS",
                    "preprocessor": "PostCSS",
                    "responsive": True,
                    "darkMode": False
                },
                "routing": {
                    "library": "React Router v6",
                    "type": "browser-history",
                    "lazyLoading": True
                },
                "stateManagement": {
                    "library": "Context API",
                    "alternative": "Zustand (if needed)"
                },
                "forms": {
                    "library": "React Hook Form",
                    "validation": "Zod"
                },
                "httpClient": {
                    "library": "Axios",
                    "baseURL": "process.env.VITE_API_URL",
                    "timeout": 10000
                },
                "icons": {
                    "library": "Lucide React",
                    "alternative": "React Icons"
                },
                "optimization": {
                    "lazyLoadComponents": True,
                    "codeSplitting": True,
                    "imageLazyLoad": True,
                    "bundleAnalyzer": True
                },
                "seo": {
                    "metaTags": True,
                    "semanticHTML": True,
                    "structuredData": True
                },
                "accessibility": {
                    "wcagLevel": "AA",
                    "ariaLabels": True,
                    "keyboardNavigation": True,
                    "screenReaderSupport": True
                },
                "testing": {
                    "unit": "Vitest",
                    "integration": "React Testing Library",
                    "e2e": "Playwright"
                }
            },
            "codeQualityStandards": {
                "formatting": {
                    "tool": "Prettier",
                    "printWidth": 100,
                    "tabWidth": 2,
                    "semi": True,
                    "singleQuote": True,
                    "trailingComma": "es5"
                },
                "linting": {
                    "tool": "ESLint",
                    "extends": ["eslint:recommended", "plugin:react/recommended", "plugin:@typescript-eslint/recommended"],
                    "rules": {
                        "noConsole": "warn",
                        "noUnusedVars": "error"
                    }
                },
                "typeChecking": {
                    "tool": "TypeScript",
                    "strict": True,
                    "noImplicitAny": True,
                    "strictNullChecks": True
                },
                "namingConventions": {
                    "components": "PascalCase",
                    "functions": "camelCase",
                    "constants": "UPPER_SNAKE_CASE",
                    "files": {
                        "components": "PascalCase.tsx",
                        "hooks": "use*.ts",
                        "utils": "camelCase.ts"
                    }
                },
                "documentation": {
                    "comments": "JSDoc for complex functions",
                    "readme": True,
                    "changelog": False
                }
            },
            "deliverables": {
                "codebase": {
                    "structure": [
                        "src/components/ - Reusable components",
                        "src/pages/ - Page components",
                        "src/layouts/ - Layout components",
                        "src/hooks/ - Custom React hooks",
                        "src/utils/ - Utility functions",
                        "src/services/ - API services",
                        "src/types/ - TypeScript types",
                        "src/assets/ - Static assets",
                        "src/styles/ - Global styles",
                        "public/ - Public assets"
                    ]
                },
                "configFiles": [
                    "package.json",
                    "tsconfig.json",
                    "vite.config.ts",
                    "tailwind.config.js",
                    "postcss.config.js",
                    ".eslintrc.json",
                    ".prettierrc"
                ],
                "documentation": [
                    "README.md - Setup and usage instructions",
                    "ARCHITECTURE.md - Architecture overview",
                    "API.md - API endpoints documentation"
                ]
            },
            "deploymentInstructions": {
                "development": {
                    "install": "npm install",
                    "run": "npm run dev",
                    "build": "npm run build",
                    "test": "npm run test"
                },
                "production": {
                    "build": "npm run build",
                    "preview": "npm run preview",
                    "platforms": ["Vercel", "Netlify", "AWS S3 + CloudFront"]
                },
                "environmentVariables": [
                    "VITE_API_URL - Backend API URL",
                    "VITE_APP_NAME - Application name",
                    "VITE_ENABLE_ANALYTICS - Enable analytics"
                ]
            }
        }
        
        return json_spec
    
    def _build_navigation_spec(self, navigation: Dict[str, Any]) -> Dict[str, Any]:
        """Build navigation specification"""
        navbar = navigation.get("navbar", {})
        footer = navigation.get("footer", {})
        
        return {
            "navbar": {
                "position": navbar.get("position", "sticky-top"),
                "backgroundColor": "design.colors.background",
                "textColor": "design.colors.text",
                "height": "64px",
                "links": navbar.get("links", []),
                "logo": {
                    "type": "text",
                    "content": "site.name",
                    "route": "/"
                },
                "callToAction": navbar.get("callToAction", {}),
                "mobileMenu": {
                    "type": "hamburger",
                    "animation": "slide-in"
                }
            },
            "footer": {
                "backgroundColor": "design.colors.primary",
                "textColor": "#FFFFFF",
                "padding": "48px 0",
                "copyright": footer.get("copyright", ""),
                "columns": footer.get("columns", []),
                "socialLinks": {
                    "enabled": True,
                    "platforms": ["facebook", "twitter", "linkedin", "github"]
                }
            },
            "breadcrumbs": {
                "enabled": True,
                "separator": "/"
            }
        }
    
    def _build_pages_spec(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build pages specification"""
        pages_spec = []
        
        for page in pages:
            page_spec = {
                "name": page.get("name", ""),
                "path": page.get("path", ""),
                "title": f"{{site.name}} - {page.get('name', '')}",
                "description": f"Page description for {page.get('name', '')}",
                "layout": "DefaultLayout",
                "meta": {
                    "robots": "index, follow",
                    "ogType": "website"
                },
                "components": []
            }
            
            # Build components
            for component in page.get("components", []):
                comp_spec = self._build_component_spec(component)
                page_spec["components"].append(comp_spec)
            
            pages_spec.append(page_spec)
        
        return pages_spec
    
    def _build_component_spec(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Build component specification"""
        comp_type = component.get("type", "")
        comp_data = component.get("data", {})
        comp_action = component.get("action", {})
        
        comp_spec = {
            "type": comp_type,
            "id": f"{comp_type.lower()}_{id(component)}",
            "data": comp_data,
            "styling": {
                "padding": "design.spacing[8]",
                "margin": "design.spacing[0]",
                "responsive": True
            },
            "animation": {
                "entrance": "fade-in",
                "duration": "design.animations.duration.base"
            }
        }
        
        # Add actions if present
        if comp_action:
            comp_spec["actions"] = comp_action
        
        # Component-specific enhancements
        if comp_type == "Hero":
            comp_spec["styling"]["minHeight"] = "500px"
            comp_spec["styling"]["backgroundGradient"] = True
            comp_spec["animation"]["entrance"] = "slide-up"
        elif comp_type == "FeatureList":
            comp_spec["styling"]["display"] = "grid"
            comp_spec["styling"]["gridColumns"] = {"mobile": 1, "tablet": 2, "desktop": 3}
            comp_spec["styling"]["gap"] = "design.spacing[6]"
        elif comp_type == "ContactForm":
            comp_spec["validation"] = {
                "library": "React Hook Form + Zod",
                "realtime": True
            }
            comp_spec["styling"]["maxWidth"] = "600px"
        
        return comp_spec
        
    def _fill_missing_fields_with_defaults(self, ba_output: Dict[str, Any], missing_fields: List[str]) -> Dict[str, Any]:
        """
        Tự động bổ sung các field thiếu với default values
        
        Args:
            ba_output: BA output hiện tại
            missing_fields: List các field còn thiếu
            
        Returns:
            BA output đã được bổ sung default values
        """
        import copy
        filled_output = copy.deepcopy(ba_output)
        
        # Default values
        defaults = {
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
                    "maxWidth": "1280px",
                    "roundedCorners": "medium"
                }
            },
            "navigation": {
                "navbar": {
                    "position": "top",
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
        
        # Deep merge với defaults
        def deep_merge(base, update):
            for key, value in update.items():
                if key not in base:
                    base[key] = value
                elif isinstance(value, dict) and isinstance(base[key], dict):
                    deep_merge(base[key], value)
                elif isinstance(value, list) and key in missing_fields:
                    # Nếu field bị thiếu và là list, thay thế bằng default
                    base[key] = value
        
        deep_merge(filled_output, defaults)
        
        # Fill specific missing fields
        if "site" not in filled_output:
            filled_output["site"] = defaults["site"]
        else:
            for field in defaults["site"]:
                if field not in filled_output["site"] or not filled_output["site"][field]:
                    filled_output["site"][field] = defaults["site"][field]
        
        if "design" not in filled_output:
            filled_output["design"] = defaults["design"]
        else:
            for field in defaults["design"]:
                if field not in filled_output["design"]:
                    filled_output["design"][field] = defaults["design"][field]
                elif isinstance(defaults["design"][field], dict):
                    for sub_field in defaults["design"][field]:
                        if sub_field not in filled_output["design"][field]:
                            filled_output["design"][field][sub_field] = defaults["design"][field][sub_field]
        
        if "navigation" not in filled_output:
            filled_output["navigation"] = defaults["navigation"]
        else:
            for key in defaults["navigation"]:
                if key not in filled_output["navigation"]:
                    filled_output["navigation"][key] = defaults["navigation"][key]
        
        if "pages" not in filled_output or not filled_output["pages"] or len(filled_output["pages"]) == 0:
            filled_output["pages"] = defaults["pages"]
        
        return filled_output
    
    def generate_json(
        self, 
        ba_output: Dict[str, Any], 
        auto_request_suggestions: bool = True,
        interactive: bool = False
    ) -> Dict[str, Any]:
        """
        Chuyển đổi BA output thành JSON specification hoàn chỉnh
        TỰ ĐỘNG BỔ SUNG DEFAULT VALUES NẾU THIẾU DATA
        
        Args:
            ba_output: JSON output từ BA Agent
            auto_request_suggestions: Tự động gọi Suggestion Agent nếu thiếu data (deprecated - luôn dùng defaults)
            interactive: True = hỏi user chọn default/custom, False = auto dùng default (deprecated - luôn dùng defaults)
            
        Returns:
            Dict: JSON specification hoàn chỉnh để đưa vào Rocket (tự động fill missing fields)
        """
        try:
            print(f"\n{self.agent_icon} {self.agent_name} bắt đầu xử lý...")
            print("=" * 60)
            
            # BƯỚC 1: Validate BA output
            print("\n📋 BƯỚC 1: Validate BA output...")
            is_valid, missing_fields = self.validate_ba_output(ba_output)
            
            if not is_valid:
                print(f"\n⚠️ PHÁT HIỆN DATA THIẾU!")
                print(f"   Số field thiếu: {len(missing_fields)}")
                print(f"   Các field thiếu: {', '.join(missing_fields[:5])}{'...' if len(missing_fields) > 5 else ''}")
                self.missing_fields = missing_fields
                
                # BƯỚC 2: TỰ ĐỘNG BỔ SUNG DEFAULT VALUES
                print(f"\n🔄 BƯỚC 2: Tự động bổ sung default values cho các field thiếu...")
                ba_output = self._fill_missing_fields_with_defaults(ba_output, missing_fields)
                
                # BƯỚC 3: Re-validate sau khi fill
                print(f"\n🔍 BƯỚC 3: Re-validate sau khi bổ sung defaults...")
                is_valid, remaining_missing = self.validate_ba_output(ba_output)
                
                if not is_valid:
                    print(f"\n⚠️ Vẫn còn {len(remaining_missing)} field thiếu sau khi fill defaults")
                    print(f"   Các field vẫn thiếu: {', '.join(remaining_missing[:5])}{'...' if len(remaining_missing) > 5 else ''}")
                    print(f"   Tiếp tục với data hiện có...")
                else:
                    print(f"\n✅ Data đã đầy đủ sau khi bổ sung defaults!")
            else:
                print(f"✅ BA output hợp lệ - Có đầy đủ data!")
            
            # BƯỚC 4: Generate JSON specification
            print(f"\n📝 BƯỚC 4: Generate JSON specification cho Rocket...")
            
            # Tạo JSON specification từ BA output (đã được fill defaults)
            json_spec = self._build_json_specification(ba_output)
            
            print(f"\n✅ JSON SPECIFICATION ĐÃ ĐƯỢC TẠO THÀNH CÔNG!")
            print(f"   Số sections: {len(json_spec)}")
            print(f"   Số pages: {len(json_spec.get('pages', []))}")
            print("=" * 60)
            
            return json_spec
            
        except Exception as e:
            print(f"❌ Lỗi khi tạo JSON specification: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _build_comprehensive_prompt(
        self, 
        site_info: Dict, 
        design_config: Dict, 
        navigation: Dict, 
        pages: List[Dict]
    ) -> str:
        """Xây dựng prompt chi tiết"""
        
        # Header
        prompt_parts = [
            "# WEBSITE DEVELOPMENT SPECIFICATION",
            "",
            "## 🎯 PROJECT OVERVIEW",
            f"Generate a complete, production-ready website for **{site_info.get('name', 'Website')}**.",
            f"- **Language:** {site_info.get('language', 'vi')}",
            f"- **Title:** {site_info.get('defaultTitle', '')}",
            f"- **Description:** {site_info.get('defaultDescription', '')}",
            "",
        ]
        
        # Design System
        prompt_parts.extend(self._build_design_section(design_config))
        
        # Navigation Structure
        prompt_parts.extend(self._build_navigation_section(navigation))
        
        # Pages and Components
        prompt_parts.extend(self._build_pages_section(pages))
        
        # Technical Requirements
        prompt_parts.extend(self._build_technical_requirements())
        
        # Code Quality Standards
        prompt_parts.extend(self._build_quality_standards())
        
        return "\n".join(prompt_parts)
    
    def _build_design_section(self, design: Dict) -> List[str]:
        """Tạo phần Design System"""
        font_config = design.get("font", {})
        layout = design.get("layout", {})
        
        return [
            "## 🎨 DESIGN SYSTEM",
            "",
            "### Color Palette",
            f"- **Primary Color:** `{design.get('primaryColor', '#0A74DA')}`",
            f"- **Secondary Color:** `{design.get('secondaryColor', '#FF6B6B')}`",
            f"- **Background:** `{design.get('backgroundColor', '#F4F7F6')}`",
            f"- **Text Color:** `{design.get('textColor', '#212121')}`",
            "",
            "### Typography",
            f"- **Primary Font:** {font_config.get('primary', 'Inter, sans-serif')}",
            f"- **Heading Font:** {font_config.get('headings', 'Roboto Slab, serif')}",
            "",
            "### Layout",
            f"- **Max Width:** {layout.get('maxWidth', '1280px')}",
            f"- **Border Radius:** {layout.get('roundedCorners', 'trung bình')} (8-12px)",
            "",
            "### Design Principles",
            "- Modern, clean, and professional aesthetics",
            "- Responsive design (mobile-first approach)",
            "- Smooth transitions and animations",
            "- Accessibility compliant (WCAG 2.1 Level AA)",
            "",
        ]
    
    def _build_navigation_section(self, navigation: Dict) -> List[str]:
        """Tạo phần Navigation Structure"""
        navbar = navigation.get("navbar", {})
        footer = navigation.get("footer", {})
        
        sections = [
            "## 🧭 NAVIGATION STRUCTURE",
            "",
            "### Navbar",
            f"- **Position:** {navbar.get('position', 'sticky-top')}",
            "- **Links:**",
        ]
        
        # Navbar links
        for link in navbar.get("links", []):
            sections.append(f"  - {link.get('label')}: `{link.get('path')}`")
        
        # CTA Button
        cta = navbar.get("callToAction", {})
        if cta:
            sections.extend([
                "",
                "- **Call-to-Action Button:**",
                f"  - Label: {cta.get('label')}",
                f"  - Action: {cta.get('action')} → {cta.get('target')}",
            ])
        
        # Footer
        sections.extend([
            "",
            "### Footer",
            f"- **Copyright:** {footer.get('copyright', '')}",
            "- **Footer Columns:**",
        ])
        
        for column in footer.get("columns", []):
            sections.append(f"  - **{column.get('title')}:**")
            for link in column.get("links", []):
                sections.append(f"    - {link.get('label')}: `{link.get('path')}`")
        
        sections.append("")
        return sections
    
    def _build_pages_section(self, pages: List[Dict]) -> List[str]:
        """Tạo phần Pages & Components chi tiết"""
        sections = [
            "## 📄 PAGES & COMPONENTS",
            "",
            "Build the following pages with their respective components:",
            "",
        ]
        
        for idx, page in enumerate(pages, 1):
            sections.extend([
                f"### Page {idx}: {page.get('name')}",
                f"**Route:** `{page.get('path')}`",
                "",
                "**Components:**",
                "",
            ])
            
            for comp_idx, component in enumerate(page.get("components", []), 1):
                sections.extend(
                    self._format_component(comp_idx, component)
                )
            
            sections.append("")
        
        return sections
    
    def _format_component(self, index: int, component: Dict) -> List[str]:
        """Format chi tiết từng component"""
        comp_type = component.get("type", "Unknown")
        comp_data = component.get("data", {})
        comp_action = component.get("action", {})
        
        lines = [
            f"#### {index}. {comp_type} Component",
            "",
        ]
        
        # Component data
        if comp_data:
            lines.append("**Data:**")
            for key, value in comp_data.items():
                if isinstance(value, dict):
                    lines.append(f"- **{key}:**")
                    for sub_key, sub_value in value.items():
                        lines.append(f"  - {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    lines.append(f"- **{key}:**")
                    for item in value:
                        if isinstance(item, dict):
                            item_str = ", ".join([f"{k}: {v}" for k, v in item.items()])
                            lines.append(f"  - {item_str}")
                        else:
                            lines.append(f"  - {item}")
                else:
                    lines.append(f"- **{key}:** {value}")
            lines.append("")
        
        # Component actions
        if comp_action:
            lines.append("**Actions:**")
            for key, value in comp_action.items():
                lines.append(f"- {key}: `{value}`")
            lines.append("")
        
        # Component requirements
        lines.extend([
            "**Requirements:**",
            f"- Implement {comp_type} with all specified data fields",
            "- Ensure responsive design across all devices",
            "- Add smooth animations and transitions",
            "- Follow accessibility best practices",
            "",
        ])
        
        return lines
    
    def _build_technical_requirements(self) -> List[str]:
        """Tạo phần Technical Requirements"""
        return [
            "## ⚙️ TECHNICAL REQUIREMENTS",
            "",
            "### Framework & Tools",
            "- **Frontend Framework:** React 18+ with TypeScript",
            "- **Styling:** Tailwind CSS or styled-components",
            "- **Routing:** React Router v6",
            "- **State Management:** Context API or Zustand (if needed)",
            "- **Forms:** React Hook Form with validation",
            "- **Icons:** React Icons or Lucide React",
            "",
            "### API Integration",
            "- Implement API calls using Axios or Fetch API",
            "- Handle loading states, errors, and success messages",
            "- Add proper error boundaries",
            "",
            "### Performance",
            "- Lazy load components and images",
            "- Optimize bundle size",
            "- Implement code splitting",
            "- Add loading skeletons for better UX",
            "",
            "### SEO & Accessibility",
            "- Semantic HTML5 markup",
            "- Proper heading hierarchy (h1-h6)",
            "- Alt text for all images",
            "- ARIA labels where needed",
            "- Meta tags for SEO",
            "",
        ]
    
    def _build_quality_standards(self) -> List[str]:
        """Tạo phần Code Quality Standards"""
        return [
            "## ✅ CODE QUALITY STANDARDS",
            "",
            "### Code Structure",
            "- Clean, modular, and reusable components",
            "- Proper TypeScript types and interfaces",
            "- Consistent naming conventions (camelCase, PascalCase)",
            "- Comprehensive comments for complex logic",
            "",
            "### Best Practices",
            "- Follow React best practices and hooks rules",
            "- Implement error handling and validation",
            "- Use environment variables for API endpoints",
            "- Add PropTypes or TypeScript interfaces",
            "",
            "### Testing",
            "- Unit tests for utility functions",
            "- Integration tests for key user flows",
            "- E2E tests for critical paths",
            "",
            "## 🚀 DELIVERABLES",
            "",
            "Generate a complete, production-ready codebase including:",
            "",
            "1. **All React components** with proper structure",
            "2. **Styling files** (CSS/SCSS/Tailwind config)",
            "3. **Routing configuration** with all pages",
            "4. **API integration layer** with error handling",
            "5. **Configuration files** (package.json, tsconfig.json, etc.)",
            "6. **README.md** with setup instructions",
            "",
            "---",
            "",
            "**Note:** This website should be fully functional, visually appealing, and ready to deploy. "
            "All components should be interactive, responsive, and follow modern web development standards.",
        ]
    
    def process_and_generate(self, ba_json_string: str) -> str:
        """
        Process JSON string từ BA Agent và tạo prompt
        
        Args:
            ba_json_string: JSON string output từ BA Agent
            
        Returns:
            str: Prompt hoàn chỉnh
        """
        print(f"{self.agent_icon} **{self.agent_name}** đang xử lý...")
        
        try:
            # Parse JSON
            ba_output = json.loads(ba_json_string)
            
            # Generate prompt
            prompt = self.generate_prompt(ba_output)
            
            # Send success message
            print(f"✅ **Prompt đã được tạo thành công!**\nĐộ dài: {len(prompt)} ký tự")
            
            return prompt
            
        except json.JSONDecodeError as e:
            error_msg = f"❌ Lỗi parse JSON: {str(e)}"
            print(error_msg)
            return ""
        except Exception as e:
            error_msg = f"❌ Lỗi không xác định: {str(e)}"
            print(error_msg)
            return ""


# Helper function để test agent độc lập
def test_prompt_generator():
    """Test function với example JSON"""
    
    example_json = {
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
                "maxWidth": "1280px",
                "roundedCorners": "trung bình"
            }
        },
        "navigation": {
            "navbar": {
                "position": "sticky-top",
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
                    }
                ]
            }
        ]
    }
    
    agent = PromptGeneratorAgent()
    prompt = agent.generate_prompt(example_json)
    print(prompt)
    
    return prompt
