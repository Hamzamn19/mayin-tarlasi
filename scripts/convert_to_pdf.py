"""
Convert IEEE paper from Markdown to PDF via HTML+WeasyPrint.
Renders LaTeX equations as SVG via matplotlib.
"""

import re
import os
import base64
import mimetypes
from io import BytesIO

import markdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import weasyprint

PAPER_MD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "ieee_paper.md")
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "ieee_paper.pdf")
MD_DIR = os.path.dirname(PAPER_MD)

# ──────────────────────────────────────────────
#  Image helpers
# ──────────────────────────────────────────────

def resolve_image_path(src):
    if src.startswith("data:"):
        return src
    if os.path.isabs(src):
        return src
    return os.path.normpath(os.path.join(MD_DIR, src))


def image_to_base64(path):
    if not os.path.exists(path):
        print(f"  WARNING: image not found: {path}")
        return None
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"

# ──────────────────────────────────────────────
#  LaTeX → SVG renderer
# ──────────────────────────────────────────────

def latex_to_svg_data_uri(latex_code, is_inline=False):
    """
    Render a LaTeX math expression as an SVG image using matplotlib's mathtext.
    Returns a base64 data URI.
    """
    try:
        fs = 13 if is_inline else 15
        fig, ax = plt.subplots(figsize=(len(latex_code) * 0.10 + 0.3, 0.45 if is_inline else 0.55))
        ax.text(0.5, 0.5, f'${latex_code}$', fontsize=fs,
                ha='center', va='center', transform=ax.transAxes,
                usetex=False, math_fontfamily='dejavusans')
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='svg', bbox_inches='tight', pad_inches=0.04,
                    transparent=True)
        plt.close(fig)
        svg_data = buf.getvalue()
        b64 = base64.b64encode(svg_data).decode('utf-8')
        return f"data:image/svg+xml;base64,{b64}"
    except Exception as e:
        print(f"  WARNING: could not render LaTeX: {latex_code[:60]}... -> {e}")
        return None

# ──────────────────────────────────────────────
#  Markdown → HTML processing
# ──────────────────────────────────────────────

def protect_math(text):
    """Replace LaTeX math blocks and inline with placeholders so markdown
    parser does not mangle them. We use matplotlib to render them later."""
    placeholders = {}
    counter = [0]

    def repl_display(m):
        counter[0] += 1
        key = f"@@MATH_DISPLAY_{counter[0]}@@"
        placeholders[key] = m.group(1).strip()
        return key

    def repl_inline(m):
        counter[0] += 1
        key = f"@@MATH_INLINE_{counter[0]}@@"
        placeholders[key] = m.group(1).strip()
        return key

    text = re.sub(r'\$\$(.*?)\$\$', repl_display, text, flags=re.DOTALL)
    text = re.sub(r'\$(.+?)\$', repl_inline, text)
    return text, placeholders


def restore_math(html, placeholders):
    """Replace placeholders with rendered SVG images."""
    for key, latex in placeholders.items():
        is_inline = "INLINE" in key
        svg_uri = latex_to_svg_data_uri(latex, is_inline=is_inline)
        if svg_uri:
            if is_inline:
                replacement = f'<span class="inline-math"><img class="math-svg" src="{svg_uri}" alt="{latex}"/></span>'
            else:
                replacement = f'<div class="equation"><img class="math-svg" src="{svg_uri}" alt="{latex}"/></div>'
        else:
            replacement = latex  # fallback
        html = html.replace(key, replacement)
    return html


def extract_abstract(body_html):
    """Extract the abstract and wrap in a column-span div."""
    m = re.search(r'<p>\*\*Abstract—(.+?)\*\*</p>', body_html, re.DOTALL)
    if m:
        abstract_text = m.group(1).strip()
        abstract_html = f'<div class="abstract-block"><span class="abstract-label">Abstract\u2014</span>{abstract_text}</div>'
        body_html = body_html.replace(m.group(0), '', 1)
        return abstract_html, body_html
    return '', body_html


def restructure_figures(html):
    """Wrap <p> containing <img> into figure divs for captions."""

    def wrap(m):
        p = m.group(0)
        # Try alt-first then src-first
        im = re.search(r'<img\s+alt="([^"]*)"\s+src="([^"]+)"\s*/?>', p)
        if im:
            alt, src = im.group(1), im.group(2)
        else:
            im = re.search(r'<img\s+src="([^"]+)"\s+alt="([^"]*)"\s*/?>', p)
            if im:
                src, alt = im.group(1), im.group(2)
            else:
                return p
        return f'<div class="figure"><img src="{src}" alt="{alt}"/><div class="caption">{alt}</div></div>'

    html = re.sub(r'<p[^>]*>\s*<img[^>]+>\s*</p>', wrap, html)
    return html


def embed_images(html):
    """Replace image src with base64 data URIs."""

    def repl(m):
        tag = m.group(0)
        sm = re.search(r'src="([^"]+)"', tag)
        if not sm:
            return tag
        src = sm.group(1)
        if src.startswith("data:"):
            return tag
        abs_path = resolve_image_path(src)
        b64 = image_to_base64(abs_path)
        if b64:
            return tag.replace(f'src="{src}"', f'src="{b64}"')
        return tag

    return re.sub(r'<img[^>]+>', repl, html)

# ──────────────────────────────────────────────
#  Main conversion
# ──────────────────────────────────────────────

def convert(md_text):
    # Extract title
    title = ""
    for line in md_text.split('\n'):
        if line.startswith('# ') and not title:
            title = line[2:].strip()
            break

    # Protect math
    md_text, math_placeholders = protect_math(md_text)

    # Convert to HTML
    md = markdown.Markdown(extensions=['extra', 'sane_lists', 'smarty', 'codehilite'])
    html_body = md.convert(md_text)

    # Extract abstract
    abstract_html, html_body = extract_abstract(html_body)

    # Restore math as SVG
    html_body = restore_math(html_body, math_placeholders)

    # Restructure figures
    html_body = restructure_figures(html_body)

    # Wrap in full HTML document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
@page {{
  size: A4;
  margin: 1.6cm 1.4cm 1.6cm 1.4cm;
  @bottom-center {{
    content: counter(page);
    font-family: 'Times New Roman', Times, serif;
    font-size: 8pt;
    color: #555;
  }}
}}

* {{ box-sizing: border-box; }}

body {{
  font-family: 'Times New Roman', Times, serif;
  font-size: 9pt;
  line-height: 1.40;
  color: #000;
  text-align: justify;
  column-count: 2;
  column-gap: 0.55cm;
  orphans: 2;
  widows: 2;
}}

/* ── Title Block ── */
.title-block {{
  column-span: all;
  text-align: center;
  margin-bottom: 0.35cm;
  padding-bottom: 0.2cm;
  border-bottom: 1.5pt solid #000;
}}
.title-block h1 {{
  font-size: 14pt; font-weight: 700; margin: 0 0 3pt 0; line-height: 1.2;
}}
.title-block .authors {{
  font-size: 8.5pt; margin: 3pt 0 1pt 0;
}}
.title-block .affil {{
  font-size: 8pt; font-style: italic; color: #333; margin: 1pt 0;
}}
.title-block .supervisor {{
  font-size: 8pt; color: #444; margin: 2pt 0;
}}

/* ── Abstract ── */
.abstract-block {{
  column-span: all;
  margin-bottom: 0.25cm;
  font-size: 8.5pt;
  line-height: 1.35;
  text-align: justify;
}}
.abstract-label {{ font-weight: 700; font-size: 8.5pt; }}

/* ── Headings ── */
h2 {{
  font-size: 9.5pt; font-weight: 700;
  margin: 0.18cm 0 0.08cm 0;
  text-transform: uppercase; letter-spacing: 0.5pt;
  border-bottom: 0.8pt solid #000; padding-bottom: 1pt;
  break-after: avoid;
}}
h3 {{
  font-size: 9pt; font-weight: 700;
  margin: 0.12cm 0 0.04cm 0;
  break-after: avoid;
}}

/* ── Paragraphs ── */
p {{ margin: 2pt 0; text-indent: 0.3cm; }}
h2 + p, h3 + p {{ text-indent: 0; }}

/* ── Tables ── */
table {{
  width: 100%; border-collapse: collapse;
  margin: 0.12cm 0; font-size: 7.2pt; break-inside: avoid;
}}
th {{
  background: #222; color: #fff; font-weight: 600;
  padding: 2.5pt 2pt; text-align: center; font-size: 6.8pt;
  text-transform: uppercase;
}}
td {{
  padding: 2pt 2.5pt; border: 0.4pt solid #999; text-align: center;
}}
td:first-child {{ text-align: left; font-weight: 500; }}
tr:nth-child(even) td {{ background: #f4f4f4; }}

/* ── Equations (SVG images) ── */
.equation {{
  text-align: center; margin: 0.15cm 0;
  break-inside: avoid; page-break-inside: avoid;
}}
.equation img.math-svg {{
  max-width: 100%; height: auto; vertical-align: middle;
}}

.inline-math img.math-svg {{
  vertical-align: middle; display: inline;
  max-height: 1.3em; width: auto;
}}

/* ── Figures ── */
.figure {{
  text-align: center;
  margin: 0.15cm 0;
  break-inside: avoid;
  page-break-inside: avoid;
}}
.figure img {{
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}}
.figure .caption {{
  font-size: 7.5pt;
  font-style: italic;
  color: #333;
  text-align: center;
  margin-top: 2pt;
  line-height: 1.3;
}}

/* ── Lists, code, etc ── */
ul, ol {{ margin: 2pt 0; padding-left: 1.2em; }}
li {{ margin-bottom: 0.5pt; text-align: justify; }}
code {{
  font-family: 'Courier New', monospace;
  font-size: 7.2pt; background: #f0f0f0; padding: 0.3pt 1.5pt;
}}
pre {{
  font-family: 'Courier New', monospace; font-size: 6.8pt;
  background: #f5f5f5; border: 0.4pt solid #ddd;
  padding: 3pt; overflow-x: auto; break-inside: avoid;
}}
hr {{ border: none; border-top: 0.6pt solid #ccc; margin: 0.15cm 0; }}
h2, h3 {{ break-after: avoid; }}

/* ── References ── */
#references + ol, #references + ul {{
  font-size: 7.5pt; line-height: 1.35; padding-left: 1.5em;
}}
#references + ol li, #references + ul li {{
  margin-bottom: 2pt; word-break: break-word;
}}
</style>
</head>
<body>

<div class="title-block">
  <h1>{title}</h1>
  <div class="authors">Hamzah Sheikh Alashrah, Yasin Deniz Zeybek, Kemal Gençer, Emir Tayanç Kavak</div>
  <div class="affil"><em>Department of Computer Engineering, Beykoz University, Istanbul, Turkey</em></div>
  <div class="supervisor"><strong>Supervisor:</strong> Assoc. Prof. Mustafa Cem Kasapbaşı</div>
</div>

{abstract_html}

{html_body}

</body>
</html>"""

    return html


def main():
    with open(PAPER_MD, 'r', encoding='utf-8') as f:
        md_content = f.read()

    print("Converting Markdown → HTML ...")
    html_content = convert(md_content)

    print("Embedding images as base64 ...")
    html_content = embed_images(html_content)

    # Save intermediate HTML
    html_path = OUTPUT_PDF.replace('.pdf', '.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"  HTML → {html_path}")

    print("Rendering PDF with WeasyPrint ...")
    doc = weasyprint.HTML(string=html_content).render()
    doc.write_pdf(OUTPUT_PDF)
    print(f"  PDF  → {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
