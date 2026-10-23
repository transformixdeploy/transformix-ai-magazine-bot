"""Transformix AI Report: premium one-column magazine rendered with WeasyPrint.

Ported from the JTGC "premium magazine" (members.jtgc.sa) and re-skinned for
Transformix. Every page carries a real photograph: each article uses its own
source image when it downloads cleanly and is large enough, then the article
page's og:image, then (if UNSPLASH_ACCESS_KEY is set) an Unsplash photo found
from the model's English ``image_query``, then the bundled AI imagery in
templates/images. All images are downloaded before rendering, so a dead
hotlink can never leave a blank frame.
"""
import glob
import io
import json
import logging
import os
import pathlib
import re
import tempfile
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
IMAGES_DIR = os.path.join(TEMPLATE_DIR, "images")

UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "").strip()
_UA = {"User-Agent": "Mozilla/5.0 (compatible; TransformixMagazine/1.0)"}

# Coarse guard against runaway model output; the exact fit is measured (see _fit_pages).
MAX_ARTICLE_WORDS = 290
MAX_EDITOR_WORDS = 190
MAX_HIGHLIGHT_WORDS = 14

# Article pages: text must end above the footer margin, i.e. by 282 mm.
TEXT_BOTTOM_MM = 282 - 3
MIN_HERO_MM, MAX_HERO_MM = 95, 165
_MM = 96 / 25.4  # WeasyPrint lays out in CSS px

# Bundled art: the generated cover/back cover and the brand marks are never
# used as article photos.
_RESERVED_IMAGES = {"cover_generated.png", "back_cover_generated.png",
                    "transformix-mark.png", "transformix-mark-white.png"}

# English photo queries per category, used when the model gave none.
_CATEGORY_QUERIES = {
    "Strategy": "business strategy meeting technology",
    "Technology": "artificial intelligence technology",
    "Ethics": "data privacy technology",
    "Industry": "data center servers",
}


def issue_date(now=None):
    return (now or datetime.now()).strftime("%B %Y")


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
def _save_image(raw, key, dest_dir, min_width):
    """Validate image bytes and save them as a JPEG. Returns a file:// URI or None."""
    from PIL import Image
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    if w < min_width or h < 400 or w / max(h, 1) > 3.2 or h / max(w, 1) > 1.6:
        return None
    img = img.convert("RGB")
    if w > 2000:
        img = img.resize((2000, int(h * 2000 / w)))
    path = os.path.join(dest_dir, f"img_{abs(hash(key))}.jpg")
    img.save(path, "JPEG", quality=86)
    return pathlib.Path(path).as_uri()


def _download_image(url, dest_dir, min_width=900):
    """Download ``url`` and return a file:// URI, or None if it is unusable."""
    if not url or not str(url).startswith("http"):
        return None
    try:
        r = requests.get(url, headers=_UA, timeout=12)
        if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
            return None
        return _save_image(r.content, url, dest_dir, min_width)
    except Exception as exc:  # noqa: BLE001 - any failure means "use a fallback"
        logger.info("image rejected (%s): %s", str(url)[:80], exc)
        return None


def _og_image(page_url):
    """Return the og:image / twitter:image of an article page, or None."""
    if not page_url or not page_url.startswith("http"):
        return None
    try:
        r = requests.get(page_url, headers=_UA, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for attrs in ({"property": "og:image"}, {"name": "twitter:image"},
                      {"property": "og:image:url"}):
            tag = soup.find("meta", attrs=attrs)
            if tag and tag.get("content", "").startswith("http"):
                return tag["content"]
    except Exception as exc:  # noqa: BLE001
        logger.info("og:image lookup failed for %s: %s", page_url[:80], exc)
    return None


class _Unsplash:
    """Landscape search with per-issue de-duplication and Unsplash's required
    download ping. Credits are returned so the page can attribute the photo."""

    def __init__(self):
        self.used = set()

    def find(self, query, dest_dir, width=None):
        if not UNSPLASH_KEY or not query:
            return None, None
        try:
            r = requests.get(
                "https://api.unsplash.com/search/photos",
                params={"query": query, "orientation": "landscape", "per_page": 15,
                        "content_filter": "high"},
                headers={"Authorization": f"Client-ID {UNSPLASH_KEY}", "Accept-Version": "v1"},
                timeout=15,
            )
            if r.status_code != 200:
                logger.warning("unsplash search %s -> %s", query, r.status_code)
                return None, None
            for photo in r.json().get("results", []):
                if photo["id"] in self.used:
                    continue
                src = photo["urls"].get("regular")
                if width and photo["urls"].get("raw"):
                    src = f"{photo['urls']['raw']}&w={width}&fit=max&q=85"
                uri = _download_image(src, dest_dir, min_width=900)
                if not uri:
                    continue
                self.used.add(photo["id"])
                try:  # API guideline: register the download.
                    requests.get(photo["links"]["download_location"],
                                 headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"}, timeout=8)
                except Exception:  # noqa: BLE001
                    pass
                return uri, f"{photo['user']['name']} / Unsplash"
        except Exception as exc:  # noqa: BLE001
            logger.warning("unsplash search failed for %s: %s", query, exc)
        return None, None


class _Bundled:
    """The AI stock images shipped in templates/images, each used once per issue."""

    def __init__(self, dest_dir):
        self.pool = []
        for path in sorted(glob.glob(os.path.join(IMAGES_DIR, "*"))):
            if os.path.basename(path) in _RESERVED_IMAGES:
                continue
            try:
                with open(path, "rb") as fh:
                    uri = _save_image(fh.read(), path, dest_dir, min_width=900)
            except Exception:  # noqa: BLE001 - unreadable file, skip it
                uri = None
            if uri:
                self.pool.append(uri)

    def take(self):
        return self.pool.pop(0) if self.pool else None


def _local_uri(name):
    path = os.path.join(IMAGES_DIR, name)
    return pathlib.Path(path).as_uri() if os.path.exists(path) else None


def _source_name(article):
    src = article.get("source")
    if isinstance(src, dict):
        return src.get("name") or ""
    return src or ""


def link_sources(data, source_articles):
    """Fill each magazine article's image_url / url / source from the news item
    it was written from (``article_index`` is the 1-based number shown to the
    model in the prompt)."""
    for art in data.get("articles", []):
        try:
            orig = source_articles[int(art.get("article_index")) - 1]
        except (TypeError, ValueError, IndexError):
            continue
        art.setdefault("url", orig.get("url"))
        if not art.get("image_url"):
            art["image_url"] = orig.get("urlToImage") or orig.get("image") or orig.get("image_url")
        real_source = _source_name(orig)
        if real_source:
            art["source"] = real_source


def attach_images(data, dest_dir):
    """Give the cover, editor's note, every article and the back cover a photo."""
    unsplash = _Unsplash()
    bundled = _Bundled(dest_dir)
    seen = set()

    # Pass 1: each article's own photo (feed image, then the page's og:image).
    for art in data.get("articles", []):
        art["photo"], art["photo_credit"] = None, ""
        for url in (art.get("image_url"), _og_image(art.get("url"))):
            if not url or url in seen:
                continue
            uri = _download_image(url, dest_dir)
            if uri:
                seen.add(url)
                art["photo"], art["photo_credit"] = uri, art.get("source") or ""
                break

    # Editor's note hero, before the article fallbacks drain the bundled pool.
    data["editor_photo"], data["editor_photo_credit"] = unsplash.find(
        "futuristic technology abstract", dest_dir)
    if not data["editor_photo"]:
        data["editor_photo"], data["editor_photo_credit"] = bundled.take(), ""

    # Pass 2: articles still without a photo.
    for art in data.get("articles", []):
        if art["photo"]:
            continue
        for q in (art.get("image_query"), _CATEGORY_QUERIES.get(art.get("category", ""))):
            uri, credit = unsplash.find(q, dest_dir)
            if uri:
                art["photo"], art["photo_credit"] = uri, credit
                break
        else:
            art["photo"] = bundled.take()  # None -> the template draws a brand panel

    # Full-bleed covers want a tall, high-resolution photo.
    for key, query, fallback in (
            ("cover", data.get("cover_image_query") or "artificial intelligence abstract", "cover_generated.png"),
            ("back", "abstract technology network dark", "back_cover_generated.png")):
        uri, credit = unsplash.find(query, dest_dir, width=1800)
        data[f"{key}_photo"] = uri or _local_uri(fallback)
        data[f"{key}_photo_credit"] = credit or ""


# --------------------------------------------------------------------------- #
# Text fitting
# --------------------------------------------------------------------------- #
def _words(html):
    return len(re.sub(r"<[^>]+>", " ", html or "").split())


def _trim_html(html, max_words):
    """Keep whole <p>/<h3> blocks until the budget runs out; end on a sentence."""
    blocks = re.findall(r"<(h3|p)[^>]*>(.*?)</\1>", html or "", flags=re.S)
    if not blocks:
        blocks = [("p", p) for p in re.split(r"\n\s*\n", re.sub(r"<[^>]+>", "", html or "")) if p.strip()]
    out, used = [], 0
    for tag, inner in blocks:
        text = re.sub(r"<[^>]+>", "", inner).strip()
        n = len(text.split())
        if not text:
            continue
        if used + n <= max_words:
            out.append(f"<{tag}>{inner.strip()}</{tag}>")
            used += n
            continue
        if tag == "p" and max_words - used > 25:
            cut = " ".join(text.split()[: max_words - used])
            end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("."))
            if end > len(cut) // 3:
                out.append(f"<p>{cut[:end + 1]}</p>")
        break
    while out and out[-1].startswith("<h3"):  # never end on a dangling subheading
        out.pop()
    return "\n".join(out)


def _trim_words(text, max_words):
    words = (text or "").split()
    return text if len(words) <= max_words else " ".join(words[:max_words]).rstrip(",;:") + "…"


def fit_content(data):
    """Normalise the model's text and trim anything far too long for its page."""
    data["title"] = re.sub(r"^\s*AI Monthly\s*[:\-–—]\s*", "", data.get("title") or "").strip()
    note = data.get("editors_note") or ""
    if "<p" not in note:
        note = "".join(f"<p>{p.strip()}</p>" for p in note.split("\n") if p.strip())
    data["editors_note"] = _trim_html(note, MAX_EDITOR_WORDS)
    for h in data.get("highlights") or []:
        h["description"] = _trim_words(h.get("description"), MAX_HIGHLIGHT_WORDS)

    for art in data.get("articles", []):
        lead_words = len((art.get("lead") or "").split())
        art["content"] = _trim_html(art.get("content", ""), MAX_ARTICLE_WORDS - lead_words)
        art["hero_mm"] = 112


def _walk(box):
    yield box
    for child in getattr(box, "children", None) or []:
        yield from _walk(child)


def _text_bottoms(html):
    """Lay articles out on endless pages and return where each one's text ends (mm)."""
    from weasyprint import CSS, HTML
    doc = HTML(string=html, base_url=TEMPLATE_DIR + "/").render(
        stylesheets=[CSS(string="@page article { size: 210mm 2000mm !important; }")])  # user-origin CSS needs !important to beat the template
    bottoms = []
    for page in doc.pages:
        for box in _walk(page._page_box):
            el = getattr(box, "element", None)
            if el is not None and box.element_tag == "div" and el.get("class") == "body":
                bottoms.append((box.position_y + box.margin_height()) / _MM)
                break
    return bottoms


def _fit_pages(data, template):
    """Size each hero photo so its article's text ends exactly at the page
    bottom; the photo absorbs the slack instead of white space. Text is only
    cut when the photo is already at its minimum height."""
    for _ in range(3):
        bottoms = _text_bottoms(template.render(**data))
        overflowing = False
        for art, bottom in zip(data["articles"], bottoms):
            hero = art["hero_mm"] + (TEXT_BOTTOM_MM - bottom)
            if hero < MIN_HERO_MM:
                # ~0.4 mm of page per word of body text, plus a margin for line breaks
                cut = int((MIN_HERO_MM - hero) / 0.4) + 8
                art["content"] = _trim_html(art["content"], max(60, _words(art["content"]) - cut))
                overflowing = True
            art["hero_mm"] = int(max(MIN_HERO_MM, min(MAX_HERO_MM, hero)))
        if not overflowing:
            return
    logger.warning("magazine fit: some articles still overflow after 3 passes")


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def render(data, output_path, source_articles=None):
    """Render the magazine JSON ``data`` to ``output_path``. Returns the path."""
    from weasyprint import HTML

    # Keep the model's output so a layout change can re-render without a new
    # (paid) generation: python render_magazine.py weekly_blogs/last_magazine.json
    try:
        json_path = os.getenv("MAGAZINE_JSON_PATH",
                              os.path.join(BASE_DIR, "weekly_blogs", "last_magazine.json"))
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({"data": data, "source_articles": source_articles or []}, fh,
                      ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not save magazine json: %s", exc)

    data = dict(data)
    data["articles"] = [dict(a) for a in data.get("articles", [])]
    data["highlights"] = [dict(h) for h in data.get("highlights") or []]
    data["date"] = issue_date()
    if source_articles:
        link_sources(data, source_articles)
    fit_content(data)
    work = tempfile.mkdtemp(prefix="mag_")
    attach_images(data, work)

    data["mark"] = _local_uri("transformix-mark.png")
    data["mark_white"] = _local_uri("transformix-mark-white.png")
    data["sources"] = sorted({a.get("source") for a in data["articles"] if a.get("source")})

    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR),
                      autoescape=select_autoescape(enabled_extensions=("html",), default=False))
    template = env.get_template("magazine.html")
    _fit_pages(data, template)
    HTML(string=template.render(**data), base_url=TEMPLATE_DIR + "/").write_pdf(output_path)
    missing = [a.get("title") for a in data["articles"] if not a.get("photo")]
    logger.info("magazine rendered: %d articles, %d without photo -> %s",
                len(data["articles"]), len(missing), output_path)
    return output_path
