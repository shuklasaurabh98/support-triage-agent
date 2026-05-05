"""
scraper.py — Corpus builder for HackerRank, Claude, and Visa support sites.

Run standalone:  python scraper.py
Reads cache if already scraped; use force=True to re-crawl.
"""

import os, json, time, hashlib, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Optional, List, Dict

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'corpus')

SOURCES: Dict[str, dict] = {
    "HackerRank": {
        "base_url": "https://support.hackerrank.com",
        "seed_urls": [
            "https://support.hackerrank.com/hc/en-us",
            "https://support.hackerrank.com/hc/en-us/categories/115000098003",
            "https://support.hackerrank.com/hc/en-us/categories/115000098023",
            "https://support.hackerrank.com/hc/en-us/categories/115000097343",
        ],
        "sitemaps": ["https://support.hackerrank.com/hc/sitemap.xml"],
        "allowed_domain": "support.hackerrank.com",
    },
    "Claude": {
        "base_url": "https://support.claude.ai",
        "seed_urls": [
            "https://support.claude.ai/hc/en-us",
            "https://support.claude.com/en",
        ],
        "sitemaps": [
            "https://support.claude.ai/hc/sitemap.xml",
            "https://support.claude.com/sitemap.xml",
        ],
        "allowed_domain": "support.claude.ai",
    },
    "Visa": {
        "base_url": "https://www.visa.co.in",
        "seed_urls": [
            "https://www.visa.co.in/support.html",
            "https://www.visa.co.in/support/consumer/lost-stolen-card.html",
            "https://www.visa.co.in/support/consumer/transaction-disputes.html",
            "https://www.visa.co.in/support/consumer/visa-security-programs.html",
        ],
        "sitemaps": [],
        "allowed_domain": "www.visa.co.in",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def extract_sitemap_urls(session: requests.Session, sitemap_url: str) -> List[str]:
    try:
        r = session.get(sitemap_url, timeout=15)
        if r.status_code != 200:
            return []
        # Try xml parser, fall back to html.parser
        try:
            soup = BeautifulSoup(r.text, 'xml')
        except Exception:
            soup = BeautifulSoup(r.text, 'html.parser')
        urls = [loc.get_text(strip=True) for loc in soup.find_all('loc')]
        # Prefer article pages
        return [u for u in urls if '/articles/' in u or '/hc/' in u or '/support' in u][:200]
    except Exception as e:
        print(f"  [sitemap] {sitemap_url}: {e}")
        return []


def get_doc_from_page(session: requests.Session, url: str, source_key: str) -> Optional[Dict]:
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, 'lxml')

        # Strip noise
        for tag in soup(['script', 'style', 'nav', 'footer', 'header',
                         'aside', 'noscript', 'form', 'iframe']):
            tag.decompose()

        # Prefer main content blocks
        main = (
            soup.find('article') or
            soup.find('main') or
            soup.find(class_=['article-body', 'hc-article-body', 'content-body',
                               'entry-content', 'page-body', 'support-article']) or
            soup.find(id=['article-content', 'main-content', 'content']) or
            soup.body
        )

        h1 = soup.find('h1')
        title = h1.get_text(strip=True) if h1 else url

        raw = main.get_text(separator='\n', strip=True) if main else ''
        lines = [l.strip() for l in raw.splitlines() if len(l.strip()) > 15]
        content = '\n'.join(lines)

        if len(content) < 100:
            return None

        return {
            "id": hashlib.md5(url.encode()).hexdigest(),
            "source": source_key,           # e.g. "HackerRank"
            "url": url,
            "title": title,
            "content": content[:10000],
        }
    except Exception as e:
        print(f"  [page] {url}: {e}")
        return None


def find_links(html: str, base_url: str, allowed_domain: str) -> List[str]:
    soup = BeautifulSoup(html, 'lxml')
    links = set()
    for a in soup.find_all('a', href=True):
        full = urljoin(base_url, a['href'].strip())
        p = urlparse(full)
        if p.netloc == allowed_domain and p.scheme in ('http', 'https'):
            links.add(full.split('#')[0].split('?')[0])
    return list(links)


def crawl_source(source_key: str, config: dict, max_pages: int = 100) -> List[Dict]:
    session = make_session()
    visited: set = set()
    docs: List[Dict] = []

    print(f"\n[scraper] ── {source_key} (max {max_pages} pages) ──")

    # Collect URLs: sitemap first, then seeds
    queue: List[str] = []
    for sm in config.get('sitemaps', []):
        sm_urls = extract_sitemap_urls(session, sm)
        print(f"  sitemap {sm}: {len(sm_urls)} URLs")
        queue.extend(sm_urls)
    queue.extend(config['seed_urls'])
    queue = list(dict.fromkeys(queue))  # deduplicate, preserve order

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        print(f"  [{len(visited):>3}/{max_pages}] {url[:95]}", end=' ')

        try:
            resp = session.get(url, timeout=20)
            if resp.status_code != 200:
                print(f"→ HTTP {resp.status_code}")
                continue

            doc = get_doc_from_page(session, url, source_key)
            if doc:
                docs.append(doc)
                print(f"→ ✓ ({len(doc['content'])} chars)")
            else:
                print("→ skip (no content)")

            for link in find_links(resp.text, config['base_url'], config['allowed_domain']):
                if link not in visited:
                    queue.append(link)

        except Exception as e:
            print(f"→ error: {e}")

        time.sleep(0.4)

    print(f"  → {source_key}: {len(docs)} docs from {len(visited)} pages visited")
    return docs


def scrape_all(max_pages_each: int = 100, force: bool = False) -> List[Dict]:
    """Scrape all 3 sources. Uses cache unless force=True."""
    os.makedirs(DATA_DIR, exist_ok=True)
    all_docs: List[Dict] = []

    for key, config in SOURCES.items():
        out_file = os.path.join(DATA_DIR, f"{key.lower()}.json")

        if os.path.exists(out_file) and not force:
            print(f"[scraper] {key}: cache hit → {out_file}")
            with open(out_file) as f:
                docs = json.load(f)
        else:
            docs = crawl_source(key, config, max_pages=max_pages_each)
            with open(out_file, 'w') as f:
                json.dump(docs, f, indent=2)
            print(f"[scraper] {key}: saved {len(docs)} docs")

        all_docs.extend(docs)

    print(f"\n[scraper] Corpus total: {len(all_docs)} documents")
    return all_docs


def load_corpus() -> List[Dict]:
    """Load cached corpus from disk."""
    all_docs: List[Dict] = []
    for key in SOURCES:
        path = os.path.join(DATA_DIR, f"{key.lower()}.json")
        if os.path.exists(path):
            with open(path) as f:
                docs = json.load(f)
            all_docs.extend(docs)
            print(f"[corpus] {key}: {len(docs)} docs loaded")
        else:
            print(f"[corpus] WARNING: no corpus for {key} — run --scrape first")
    return all_docs


def ingest_manual(source_key: str, title: str, url: str, content: str):
    """Manually add a document when scraping is blocked."""
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = os.path.join(DATA_DIR, f"{source_key.lower()}.json")
    existing = []
    if os.path.exists(out_file):
        with open(out_file) as f:
            existing = json.load(f)
    existing.append({
        "id": hashlib.md5(url.encode()).hexdigest(),
        "source": source_key,
        "url": url,
        "title": title,
        "content": content[:10000],
    })
    with open(out_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"[corpus] Ingested '{title}' → {source_key}")


if __name__ == '__main__':
    scrape_all(force=True)
