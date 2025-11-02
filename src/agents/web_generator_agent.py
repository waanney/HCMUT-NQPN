"""
Web Generator Agent - Mô phỏng chức năng rocket.new
Sử dụng OpenAI SDK để sinh code web application từ JSON specification
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from openai import OpenAI

# Load environment variables
try:
    from dotenv import load_dotenv
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(current_dir, '..', '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """Kết quả sinh code"""
    html: str
    css: str
    javascript: str
    react_components: Optional[List[str]] = None
    package_json: Optional[str] = None
    readme: Optional[str] = None


class WebGeneratorAgent:
    """
    Agent sinh code web application từ JSON specification
    Mô phỏng chức năng của rocket.new bằng OpenAI GPT-4
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Khởi tạo Web Generator Agent
        
        Args:
            api_key: OpenAI API key (nếu None sẽ lấy từ env)
            model: Model sử dụng (gpt-4o, gpt-4o-mini, gpt-4-turbo)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        # Set timeout to 120 seconds (2 minutes) for web generation API calls
        self.openai_client = OpenAI(
            api_key=self.api_key,
            timeout=120.0  # 120 seconds timeout for API calls
        )
        logger.info(f"WebGeneratorAgent initialized with model: {model}, timeout: 120s")
    
    def generate_web_app(
        self, 
        spec: Dict[str, Any],
        framework: str = "react",
        include_backend: bool = False
    ) -> GeneratedCode:
        """
        Sinh code web application từ JSON specification
        
        Args:
            spec: JSON specification của website
            framework: Framework sử dụng (react, vue, vanilla, nextjs)
            include_backend: Có sinh backend code không
            
        Returns:
            GeneratedCode object chứa toàn bộ code đã sinh
        """
        logger.info(f"Generating web app with framework: {framework}")
        
        # Validate spec
        if not spec or not isinstance(spec, dict):
            raise ValueError("Invalid specification format")
        
        # Sinh HTML/CSS/JS cơ bản
        html = self._generate_html(spec, framework)
        css = self._generate_css(spec)
        javascript = self._generate_javascript(spec, framework)
        
        # Nếu là React/Vue, sinh components
        react_components = None
        package_json = None
        
        if framework in ["react", "vue", "nextjs"]:
            react_components = self._generate_react_components(spec, framework)
            package_json = self._generate_package_json(spec, framework)
        
        # Sinh README
        readme = self._generate_readme(spec, framework)
        
        result = GeneratedCode(
            html=html,
            css=css,
            javascript=javascript,
            react_components=react_components,
            package_json=package_json,
            readme=readme
        )
        
        logger.info("Web app generation completed")
        return result
    
    def _generate_html(self, spec: Dict[str, Any], framework: str) -> str:
        """Sinh HTML code từ specification"""
        
        # Extract design info for better prompting
        design = spec.get("design", {})
        fonts = design.get("font", {})
        
        prompt = f"""
You are an expert web developer. Generate a COMPLETE, PROFESSIONAL, PRODUCTION-READY website that looks like a REAL business website.

SPECIFICATION:
{json.dumps(spec, indent=2, ensure_ascii=False)}

🎯 CRITICAL REQUIREMENTS:

1. TECHNICAL SETUP:
   - MUST link: <link rel="stylesheet" href="styles.css">
   - MUST import Google Fonts: <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Slab:wght@400;700&display=swap" rel="stylesheet">
   - Use semantic CSS classes (NO Tailwind utility classes)
   - Add favicon link: <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚀</text></svg>">

2. COMPLETE SECTIONS (Generate REAL content for each):
   
   a) HEADER/NAVBAR:
      - Logo (text or emoji: 🚀 {spec.get('site', {}).get('name', 'Company')})
      - Navigation links (Home, Services, About, Contact)
      - CTA button (highlighted, different color)
      - Mobile hamburger menu icon (☰)
   
   b) HERO SECTION:
      - Large heading with company tagline
      - Descriptive subtitle (2-3 sentences about the business)
      - 2 CTA buttons (primary & secondary)
      - Background gradient or image placeholder
      - Stats bar (e.g., "500+ Clients | 10+ Years | 24/7 Support")
   
   c) FEATURES/SERVICES SECTION:
      - Section title: "Our Services" or "What We Offer"
      - 3-6 feature cards with:
         * Large emoji icons (🤖 ☁️ 🔒 💡 📊 🎯)
         * Feature title
         * 2-3 sentences describing the feature
         * "Learn More →" link
      - Grid layout (responsive)
   
   d) ABOUT US SECTION:
      - Section title: "About {spec.get('site', {}).get('name', 'Us')}"
      - 2-3 paragraphs about company mission, vision, values
      - Include specific details like:
         * "Founded in 2020..."
         * "Our team of 50+ experts..."
         * "Serving clients across 30+ countries..."
      - Team stats or achievements
      - Optional: Image placeholder or emoji (👥 🌍 💼)
   
   e) TESTIMONIALS/SOCIAL PROOF:
      - Section title: "What Our Clients Say"
      - 3 testimonial cards with:
         * Quote (2-3 sentences)
         * Client name & company
         * Star rating (⭐⭐⭐⭐⭐)
         * Profile emoji (👨‍💼 👩‍💻 👔)
   
   f) CONTACT SECTION:
      - Section title: "Get In Touch"
      - Contact form with fields:
         * Full Name (required)
         * Email (required)
         * Phone (optional)
         * Message (textarea)
         * Submit button
      - Contact info sidebar:
         * 📧 Email: contact@company.com
         * 📞 Phone: +1 (555) 123-4567
         * 📍 Address: 123 Business St, City, Country
         * 🕐 Hours: Mon-Fri 9AM-6PM
   
   g) FOOTER:
      - 3-4 columns with:
         * Company info & logo
         * Quick links (About, Services, Privacy, Terms)
         * Social media icons (Facebook, Twitter, LinkedIn, GitHub)
         * Newsletter signup form
      - Bottom bar with copyright & additional links
      - Background: Primary color with white text

3. DESIGN REQUIREMENTS:
   - Use REAL emoji icons throughout (🚀 💼 📊 🎯 ⭐ 📧 etc.)
   - Proper spacing (section padding: 80px-120px)
   - Box shadows on cards (0 4px 6px rgba(0,0,0,0.1))
   - Rounded corners (8px-12px)
   - Hover effects (transform, shadow, color transitions)
   - Smooth scroll behavior
   - Consistent typography hierarchy (h1: 3rem, h2: 2rem, h3: 1.5rem)

4. CONTENT QUALITY:
   - Write REALISTIC, PROFESSIONAL content (not placeholder text)
   - Use industry-specific terminology
   - Include specific numbers and metrics
   - Make it sound like a real business website
   - Minimum 150-200 words of actual content

5. COLORS TO USE:
   - Primary: {design.get('primaryColor', '#0A74DA')}
   - Secondary: {design.get('secondaryColor', '#FF6B6B')}
   - Background: {design.get('backgroundColor', '#F4F7F6')}
   - Text: {design.get('textColor', '#212121')}
   - Accent: Use lighter/darker shades of primary

6. RESPONSIVE DESIGN:
   - Mobile-first approach
   - Breakpoints: 768px (tablet), 1024px (desktop)
   - Hamburger menu on mobile
   - Stacked layout on small screens

7. SEO & ACCESSIBILITY:
   - Semantic HTML5 tags (header, nav, main, section, article, footer)
   - Alt text for images (use descriptive text)
   - ARIA labels where needed
   - Meta description, keywords, OG tags
   - Proper heading hierarchy (only one h1)

OUTPUT: Return a COMPLETE, BEAUTIFUL, PROFESSIONAL HTML page that looks like a real production website. Include ALL sections above with REAL content.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an HTML code generator. You ONLY output valid HTML code. NO explanations, NO markdown, NO text before or after the code. Start IMMEDIATELY with <!DOCTYPE html> and end with </html>. NEVER include phrases like 'Here is' or 'Here's a'. ONLY CODE."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=4000
            )
            
            html_code = response.choices[0].message.content.strip()
            
            # Clean up response - remove explanatory text
            # Find the actual HTML start
            doctype_index = html_code.find('<!DOCTYPE')
            if doctype_index > 0:
                html_code = html_code[doctype_index:]
            
            # Remove markdown code blocks if present
            if html_code.startswith("```"):
                html_code = html_code.split("```")[1]
                if html_code.startswith("html"):
                    html_code = html_code[4:]
                html_code = html_code.strip()
            
            # Remove any trailing markdown or text after </html>
            html_end = html_code.find('</html>')
            if html_end != -1:
                html_code = html_code[:html_end + 7]
            
            return html_code.strip()
            
        except Exception as e:
            logger.error(f"Error generating HTML: {e}")
            return self._get_fallback_html(spec)
    
    def _generate_css(self, spec: Dict[str, Any]) -> str:
        """Sinh CSS code từ specification"""
        
        design = spec.get("design", {})
        colors = design.get("colors", design)  # Support both formats
        fonts = design.get("typography", {}).get("fontFamily", design.get("font", {}))
        
        prompt = f"""
You are a professional CSS designer. Generate CLEAN, MODERN, MINIMAL CSS that makes websites look professional.

DESIGN SPEC:
{json.dumps(design, indent=2, ensure_ascii=False)}

🎨 EXACT CSS STRUCTURE:

/* CSS Variables */
:root {{
  --primary: {colors.get('primary', colors.get('primaryColor', '#0A74DA'))};
  --primary-dark: #0856a8;
  --secondary: {colors.get('secondary', colors.get('secondaryColor', '#FF6B6B'))};
  --bg: {colors.get('background', colors.get('backgroundColor', '#F4F7F6'))};
  --text: {colors.get('text', colors.get('textColor', '#212121'))};
  --text-light: #666;
  --white: #fff;
  --gray: #f8f9fa;
  
  --font-body: {fonts.get('primary', "'Inter', sans-serif")};
  --font-heading: {fonts.get('headings', "'Poppins', sans-serif")};
  
  --shadow: 0 2px 8px rgba(0,0,0,0.1);
  --shadow-lg: 0 8px 24px rgba(0,0,0,0.12);
  --radius: 12px;
  --transition: 0.3s ease;
  
  --container: {design.get('layout', {}).get('maxWidth', '1200px')};
}}

/* Reset & Base */
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

html {{ scroll-behavior: smooth; }}

body {{
  font-family: var(--font-body);
  color: var(--text);
  background: var(--bg);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}}

/* Typography */
h1, h2, h3, h4 {{
  font-family: var(--font-heading);
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: 1rem;
}}

h1 {{ font-size: clamp(2rem, 5vw, 3.5rem); }}
h2 {{ font-size: clamp(1.75rem, 4vw, 2.5rem); }}
h3 {{ font-size: 1.5rem; }}
h4 {{ font-size: 1.25rem; }}

p {{ margin-bottom: 1rem; line-height: 1.7; }}

a {{ text-decoration: none; color: inherit; }}

/* Container */
.container {{
  max-width: var(--container);
  margin: 0 auto;
  padding: 0 20px;
}}

/* Navbar */
.navbar {{
  position: sticky;
  top: 0;
  background: var(--white);
  box-shadow: var(--shadow);
  padding: 1rem 0;
  z-index: 100;
}}

.navbar-content {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 2rem;
}}

.logo {{
  font-size: 1.5rem;
  font-weight: 700;
  font-family: var(--font-heading);
}}

.nav-menu {{
  display: flex;
  gap: 2rem;
}}

.nav-menu a {{
  color: var(--text);
  font-weight: 500;
  transition: var(--transition);
}}

.nav-menu a:hover {{
  color: var(--primary);
}}

/* Hero */
.hero {{
  background: linear-gradient(135deg, var(--primary), var(--secondary));
  color: var(--white);
  padding: 100px 20px;
  text-align: center;
}}

.hero h1 {{
  margin-bottom: 1rem;
  color: var(--white);
}}

.hero-subtitle {{
  font-size: 1.25rem;
  margin-bottom: 2rem;
  opacity: 0.95;
}}

.hero-buttons {{
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin-bottom: 3rem;
}}

.stats {{
  display: flex;
  gap: 3rem;
  justify-content: center;
  flex-wrap: wrap;
}}

.stat-item {{
  font-size: 1.1rem;
}}

.stat-item strong {{
  display: block;
  font-size: 2rem;
  font-weight: 700;
}}

/* Buttons */
.btn {{
  display: inline-block;
  padding: 0.875rem 2rem;
  border-radius: var(--radius);
  font-weight: 600;
  transition: var(--transition);
  cursor: pointer;
  border: none;
  font-size: 1rem;
}}

.btn-primary {{
  background: var(--primary);
  color: var(--white);
}}

.btn-primary:hover {{
  background: var(--primary-dark);
  transform: translateY(-2px);
  box-shadow: var(--shadow-lg);
}}

.btn-secondary {{
  background: transparent;
  color: var(--white);
  border: 2px solid var(--white);
}}

.btn-secondary:hover {{
  background: var(--white);
  color: var(--primary);
}}

/* Sections */
section {{
  padding: 80px 20px;
}}

.section-header {{
  text-align: center;
  margin-bottom: 3rem;
}}

.section-header p {{
  color: var(--text-light);
  font-size: 1.125rem;
}}

/* Features */
.feature-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}}

.feature-card {{
  background: var(--white);
  padding: 2rem;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  transition: var(--transition);
}}

.feature-card:hover {{
  transform: translateY(-8px);
  box-shadow: var(--shadow-lg);
}}

.feature-icon {{
  font-size: 3.5rem;
  margin-bottom: 1rem;
  display: block;
}}

.feature-link {{
  color: var(--primary);
  font-weight: 600;
  display: inline-block;
  margin-top: 1rem;
}}

/* About */
.about {{
  background: var(--white);
}}

.about-content {{
  max-width: 800px;
  margin: 0 auto;
}}

.about-stats {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 2rem;
  margin-top: 3rem;
  text-align: center;
}}

.stat-number {{
  font-size: 2.5rem;
  font-weight: 700;
  color: var(--primary);
}}

.stat-label {{
  color: var(--text-light);
}}

/* Testimonials */
.testimonial-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
}}

.testimonial-card {{
  background: var(--white);
  padding: 2rem;
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  border-left: 4px solid var(--primary);
}}

.stars {{
  color: #ffc107;
  font-size: 1.25rem;
  margin-bottom: 1rem;
}}

.testimonial-text {{
  font-style: italic;
  margin-bottom: 1.5rem;
  color: var(--text-light);
}}

.testimonial-author strong {{
  display: block;
  color: var(--text);
  margin-bottom: 0.25rem;
}}

.testimonial-author span {{
  color: var(--text-light);
  font-size: 0.9rem;
}}

/* Contact */
.contact {{
  background: var(--white);
}}

.contact-wrapper {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
}}

.contact-form {{
  display: flex;
  flex-direction: column;
  gap: 1rem;
}}

.contact-form input,
.contact-form textarea {{
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: var(--radius);
  font-family: var(--font-body);
  font-size: 1rem;
}}

.contact-form input:focus,
.contact-form textarea:focus {{
  outline: none;
  border-color: var(--primary);
}}

.contact-info {{
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}}

.info-item {{
  display: flex;
  gap: 1rem;
  align-items: start;
}}

.info-icon {{
  font-size: 1.5rem;
}}

/* Footer */
.footer {{
  background: #1a1a1a;
  color: var(--white);
  padding: 60px 20px 20px;
}}

.footer-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  margin-bottom: 2rem;
}}

.footer-col h4 {{
  margin-bottom: 1rem;
  color: var(--white);
}}

.footer-col ul {{
  list-style: none;
}}

.footer-col ul li {{
  margin-bottom: 0.5rem;
}}

.footer-col a {{
  color: rgba(255,255,255,0.7);
  transition: var(--transition);
}}

.footer-col a:hover {{
  color: var(--white);
}}

.social-links {{
  display: flex;
  gap: 1rem;
  font-size: 1.5rem;
}}

.footer-bottom {{
  text-align: center;
  padding-top: 2rem;
  border-top: 1px solid rgba(255,255,255,0.1);
  color: rgba(255,255,255,0.6);
}}

/* Responsive */
@media (max-width: 768px) {{
  .navbar-content {{ flex-direction: column; gap: 1rem; }}
  .nav-menu {{ flex-direction: column; text-align: center; }}
  .hero {{ padding: 60px 20px; }}
  .hero-buttons {{ flex-direction: column; }}
  section {{ padding: 60px 20px; }}
  .contact-wrapper {{ grid-template-columns: 1fr; }}
  .footer-grid {{ grid-template-columns: 1fr; }}
}}

🚨 CRITICAL OUTPUT FORMAT:
- Start IMMEDIATELY with: :root {{ or /* CSS */ or * {{
- NO text before the CSS code
- NO markdown code blocks (```)
- NO explanations like "Here's the CSS..." or "This CSS..."
- NO instructions or text after the CSS
- ONLY valid CSS code

REMEMBER: Your response must START with CSS code - nothing else!
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a CSS code generator. You ONLY output valid CSS code. NO explanations, NO markdown, NO text. Start IMMEDIATELY with :root or /* comment */ or * {. NEVER include phrases like 'Here is' or instructions. ONLY CODE."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_completion_tokens=4000
            )
            
            css_code = response.choices[0].message.content.strip()
            
            # Clean up - find actual CSS start
            # Look for :root or * selector
            root_index = css_code.find(':root')
            star_index = css_code.find('* {')
            comment_index = css_code.find('/*')
            
            start_indices = [i for i in [root_index, star_index, comment_index] if i != -1]
            if start_indices:
                css_code = css_code[min(start_indices):]
            
            # Remove markdown code blocks if present
            if css_code.startswith("```"):
                css_code = css_code.split("```")[1]
                if css_code.startswith("css"):
                    css_code = css_code[3:]
                css_code = css_code.strip()
            
            # Remove any trailing markdown
            if '```' in css_code:
                css_code = css_code.split('```')[0]
            
            return css_code.strip()
            
        except Exception as e:
            logger.error(f"Error generating CSS: {e}")
            return self._get_fallback_css(spec)
    
    def _generate_javascript(self, spec: Dict[str, Any], framework: str) -> str:
        """Sinh JavaScript code từ specification"""
        
        pages = spec.get("pages", [])
        navigation = spec.get("navigation", {})
        
        prompt = f"""
You are a JavaScript expert. Generate JavaScript code from the following specification:

FRAMEWORK: {framework}

PAGES:
{json.dumps(pages, indent=2, ensure_ascii=False)}

NAVIGATION:
{json.dumps(navigation, indent=2, ensure_ascii=False)}

REQUIREMENTS:
1. Handle navigation (SPA routing if React/Vue)
2. Handle form submissions
3. Handle modals/popups
4. Smooth scrolling
5. Interactive components
6. API calls for forms
7. Modern ES6+ syntax
8. Error handling

OUTPUT: Only return the complete JavaScript code, without explanations.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert JavaScript developer. Generate clean, efficient, and modern JavaScript code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                
                max_completion_tokens=4000
            )
            
            js_code = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if js_code.startswith("```"):
                js_code = js_code.split("```")[1]
                if js_code.startswith("javascript") or js_code.startswith("js"):
                    js_code = js_code[10:] if js_code.startswith("javascript") else js_code[2:]
                js_code = js_code.strip()
            
            return js_code
            
        except Exception as e:
            logger.error(f"Error generating JavaScript: {e}")
            return self._get_fallback_javascript(spec)
    
    def _generate_react_components(self, spec: Dict[str, Any], framework: str) -> List[str]:
        """Sinh React/Vue components từ specification"""
        
        pages = spec.get("pages", [])
        components_list = []
        
        # Trích xuất tất cả component types
        component_types = set()
        for page in pages:
            for component in page.get("components", []):
                component_types.add(component.get("type"))
        
        prompt = f"""
You are a {framework} expert. Generate the following components:

FRAMEWORK: {framework}
COMPONENT TYPES: {list(component_types)}

FULL SPECIFICATION:
{json.dumps(spec, indent=2, ensure_ascii=False)}

REQUIREMENTS:
1. Create a separate component for each type
2. Use modern {framework} best practices
3. Functional components with hooks (if React)
4. Props validation
5. Responsive design
6. Accessibility
7. Clean code with comments

OUTPUT: Return a JSON array with the format:
[
  {{
    "name": "ComponentName",
    "code": "// component code here"
  }}
]

Only return the JSON array, without explanations.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are an expert {framework} developer. Generate production-ready components."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                
                max_completion_tokens=6000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if result.startswith("```"):
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]
                result = result.strip()
            
            components_list = json.loads(result)
            return components_list
            
        except Exception as e:
            logger.error(f"Error generating components: {e}")
            return []
    
    def _generate_package_json(self, spec: Dict[str, Any], framework: str) -> str:
        """Sinh package.json cho React/Vue project"""
        
        site_name = spec.get("site", {}).get("name", "my-app")
        
        prompt = f"""
Create a package.json for a {framework} project with the following information:

SITE NAME: {site_name}
FRAMEWORK: {framework}

REQUIREMENTS:
1. Include all necessary dependencies for {framework}
2. Scripts: start, build, test, deploy
3. Modern dependencies (latest stable versions)
4. Development dependencies
5. ESLint, Prettier configs

OUTPUT: Only return the package.json content, without explanations.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in JavaScript package management."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                
                max_completion_tokens=2000
            )
            
            package_json = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if package_json.startswith("```"):
                package_json = package_json.split("```")[1]
                if package_json.startswith("json"):
                    package_json = package_json[4:]
                package_json = package_json.strip()
            
            return package_json
            
        except Exception as e:
            logger.error(f"Error generating package.json: {e}")
            return self._get_fallback_package_json(framework)
    
    def _generate_readme(self, spec: Dict[str, Any], framework: str) -> str:
        """Sinh README.md cho project"""
        
        site_info = spec.get("site", {})
        
        prompt = f"""
Create a README.md for a web project with the following information:

SITE NAME: {site_info.get('name', 'Website')}
DESCRIPTION: {site_info.get('defaultDescription', 'A web application')}
FRAMEWORK: {framework}
LANGUAGE: {site_info.get('language', 'en')}

REQUIREMENTS:
1. Project description
2. Features list
3. Installation instructions
4. How to run locally
5. How to build for production
6. Project structure
7. Technologies used
8. License

OUTPUT: Only return the README.md content with Markdown format.
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical writer."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                
                max_completion_tokens=2000
            )
            
            readme = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if readme.startswith("```"):
                readme = readme.split("```")[1]
                if readme.startswith("markdown") or readme.startswith("md"):
                    readme = readme[8:] if readme.startswith("markdown") else readme[2:]
                readme = readme.strip()
            
            return readme
            
        except Exception as e:
            logger.error(f"Error generating README: {e}")
            return f"# {site_info.get('name', 'Project')}\n\n{site_info.get('defaultDescription', '')}"
    
    def _get_fallback_html(self, spec: Dict[str, Any]) -> str:
        """HTML fallback khi API call thất bại"""
        site = spec.get("site", {})
        return f"""<!DOCTYPE html>
<html lang="{site.get('language', 'en')}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{site.get('defaultTitle', 'Website')}</title>
    <meta name="description" content="{site.get('defaultDescription', '')}">
</head>
<body>
    <h1>{site.get('name', 'Website')}</h1>
    <p>{site.get('defaultDescription', '')}</p>
</body>
</html>"""
    
    def _get_fallback_css(self, spec: Dict[str, Any]) -> str:
        """CSS fallback khi API call thất bại"""
        design = spec.get("design", {})
        return f"""
:root {{
    --primary-color: {design.get('primaryColor', '#0A74DA')};
    --secondary-color: {design.get('secondaryColor', '#FF6B6B')};
    --background-color: {design.get('backgroundColor', '#F4F7F6')};
    --text-color: {design.get('textColor', '#212121')};
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: {design.get('font', {}).get('primary', 'sans-serif')};
    background-color: var(--background-color);
    color: var(--text-color);
}}
"""
    
    def _get_fallback_javascript(self, spec: Dict[str, Any]) -> str:
        """JavaScript fallback khi API call thất bại"""
        return """
// Basic interactive functionality
document.addEventListener('DOMContentLoaded', function() {
    console.log('Website loaded');
});
"""
    
    def _get_fallback_package_json(self, framework: str) -> str:
        """package.json fallback khi API call thất bại"""
        return json.dumps({
            "name": "my-web-app",
            "version": "1.0.0",
            "scripts": {
                "start": "npm start",
                "build": "npm run build"
            },
            "dependencies": {}
        }, indent=2)
    
    def save_generated_code(self, code: GeneratedCode, output_dir: str):
        """
        Lưu generated code vào files
        
        Args:
            code: GeneratedCode object
            output_dir: Thư mục output
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Lưu HTML
        with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(code.html)
        
        # Lưu CSS
        with open(os.path.join(output_dir, "styles.css"), "w", encoding="utf-8") as f:
            f.write(code.css)
        
        # Lưu JavaScript
        with open(os.path.join(output_dir, "script.js"), "w", encoding="utf-8") as f:
            f.write(code.javascript)
        
        # Lưu React components nếu có
        if code.react_components:
            components_dir = os.path.join(output_dir, "components")
            os.makedirs(components_dir, exist_ok=True)
            
            for component in code.react_components:
                component_name = component.get("name", "Component")
                component_code = component.get("code", "")
                
                with open(os.path.join(components_dir, f"{component_name}.jsx"), "w", encoding="utf-8") as f:
                    f.write(component_code)
        
        # Lưu package.json nếu có
        if code.package_json:
            with open(os.path.join(output_dir, "package.json"), "w", encoding="utf-8") as f:
                f.write(code.package_json)
        
        # Lưu README nếu có
        if code.readme:
            with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
                f.write(code.readme)
        
        logger.info(f"Code saved to: {output_dir}")


# Test code
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Sample specification (từ yêu cầu của user)
    sample_spec = {
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
                    },
                    {
                        "title": "Pháp lý",
                        "links": [
                            {"label": "Chính sách", "path": "/privacy"},
                            {"label": "Điều khoản", "path": "/terms"}
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
                                {"name": "Giải pháp AI", "icon": "ai-icon"},
                                {"name": "Cloud Hosting", "icon": "cloud-icon"},
                                {"name": "Bảo mật mạng", "icon": "security-icon"}
                            ]
                        }
                    }
                ]
            },
            {
                "name": "Trang liên hệ",
                "path": "/contact",
                "components": [
                    {
                        "type": "ContactForm",
                        "data": {
                            "title": "Gửi tin nhắn cho chúng tôi",
                            "fields": [
                                {"name": "name", "label": "Họ và tên", "type": "text"},
                                {"name": "email", "label": "Email", "type": "email"}
                            ]
                        },
                        "action": {
                            "onSubmit": "apiCall",
                            "target": "https://api.nexustech.com/contact"
                        }
                    }
                ]
            }
        ]
    }
    
    print("🚀 Web Generator Agent - Test Run")
    print("=" * 60)
    
    try:
        # Initialize agent
        agent = WebGeneratorAgent()
        
        print("\n📝 Generating web application code...")
        print(f"Site: {sample_spec['site']['name']}")
        print(f"Framework: React")
        print()
        
        # Generate code
        result = agent.generate_web_app(
            spec=sample_spec,
            framework="react",
            include_backend=False
        )
        
        print("✅ Code generation completed!\n")
        
        # Display results
        print("=" * 60)
        print("📄 HTML Preview (first 500 chars):")
        print("-" * 60)
        print(result.html[:500] + "...")
        print()
        
        print("=" * 60)
        print("🎨 CSS Preview (first 500 chars):")
        print("-" * 60)
        print(result.css[:500] + "...")
        print()
        
        print("=" * 60)
        print("⚡ JavaScript Preview (first 500 chars):")
        print("-" * 60)
        print(result.javascript[:500] + "...")
        print()
        
        if result.react_components:
            print("=" * 60)
            print(f"⚛️  React Components Generated: {len(result.react_components)}")
            print("-" * 60)
            for comp in result.react_components:
                print(f"  - {comp.get('name', 'Unknown')}")
        
        print("\n" + "=" * 60)
        print("💾 Saving generated code to output directory...")
        
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "generated_website")
        agent.save_generated_code(result, output_dir)
        
        print(f"✅ Code saved to: {output_dir}")
        print("\n🎉 Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
