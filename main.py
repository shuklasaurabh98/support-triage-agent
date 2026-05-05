"""
main.py — Terminal entrypoint for the Multi-Domain Support Triage Agent.

Commands:
  python main.py --scrape          Crawl support sites → corpus JSON
  python main.py --build-index     Build TF-IDF index from corpus
  python main.py --run             Process support_tickets.csv → output.csv
  python main.py --interactive     Interactive single-ticket mode
  python main.py --all             Scrape + index + run (full pipeline)

Environment:
  OLLAMA_BASE_URL     Optional, defaults to http://localhost:11434
  OLLAMA_MODEL        Optional, defaults to llama3.1:8b
"""

import os, sys, csv, json, argparse, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE        = Path(__file__).parent
DATA_DIR    = BASE / "data"
CORPUS_DIR  = DATA_DIR / "corpus"
ISSUES_DIR  = DATA_DIR / "support_issues"
LOGS_DIR    = BASE / "logs"

INPUT_CSV   = ISSUES_DIR / "support_ticket.csv"           # production tickets
LEGACY_CSV  = ISSUES_DIR / "support_tickets.csv"          # alternate plural name
FALLBACK_CSV = ISSUES_DIR / "support_issues.csv"          # local/default tickets
SAMPLE_CSV  = ISSUES_DIR / "sample_support_tickets.csv"   # with expected outputs
OUTPUT_CSV  = ISSUES_DIR / "output.csv"
LOG_FILE    = LOGS_DIR / "log.txt"

# Required output columns
OUTPUT_COLS = ["ticket_id", "status", "product_area", "response", "justification", "request_type"]


# ── Logger ────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, 'a', encoding='utf-8')
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        self._write(f"\n{'='*70}\nSESSION {ts}\n{'='*70}")

    def _write(self, msg: str):
        self._f.write(msg + "\n")
        self._f.flush()

    def log(self, msg: str, print_too: bool = True):
        self._write(msg)
        if print_too:
            print(msg)

    def log_result(self, ticket_id: str, issue: str, result: dict):
        self._write(f"\n[Ticket {ticket_id}]")
        self._write(f"ISSUE: {issue[:300]}")
        self._write(f"STATUS:       {result.get('status')}")
        self._write(f"PRODUCT AREA: {result.get('product_area')}")
        self._write(f"REQUEST TYPE: {result.get('request_type')}")
        self._write(f"JUSTIFICATION: {result.get('justification')}")
        self._write(f"RESPONSE:\n{result.get('response')}")
        self._write("─" * 60)

    def close(self):
        self._f.close()


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_tickets(path: Path) -> list:
    """Load tickets; normalise column names to lowercase."""
    if not path.exists():
        if path == INPUT_CSV:
            for candidate in (LEGACY_CSV, FALLBACK_CSV):
                if candidate.exists():
                    print(f"[main] {path.name} not found — using {candidate.name} instead")
                    path = candidate
                    break
            else:
                print(f"[main] ✗ File not found: {path}")
                sys.exit(1)
        else:
            print(f"[main] ✗ File not found: {path}")
            sys.exit(1)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Normalise headers
    normalised = []
    for row in rows:
        normalised.append({k.strip().lower(): v.strip() for k, v in row.items()})
    print(f"[main] Loaded {len(normalised)} tickets from {path.name}")
    return normalised


def get_field(row: dict, *candidates: str, default: str = "") -> str:
    for c in candidates:
        if c in row and row[c]:
            return row[c]
    return default


def write_output(results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLS, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            writer.writerow({c: r.get(c, '') for c in OUTPUT_COLS})
    print(f"\n[main] ✓ Output written → {path}")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def do_scrape(force: bool = False):
    from scraper import scrape_all
    print("[main] Starting corpus scrape…")
    docs = scrape_all(max_pages_each=100, force=force)
    print(f"[main] Scrape done — {len(docs)} total docs")
    return docs


def do_build_index(docs=None):
    from scraper import load_corpus
    from retriever import CorpusRetriever
    if docs is None:
        docs = load_corpus()
    if not docs:
        print("[main] ✗ No corpus. Run --scrape first.")
        sys.exit(1)
    r = CorpusRetriever()
    r.build_index(docs)
    return r


def get_retriever():
    from retriever import get_retriever as _gr
    from scraper import load_corpus
    try:
        return _gr()
    except RuntimeError:
        print("[main] No cached index — building from corpus…")
        docs = load_corpus()
        return _gr(docs)


def do_run(logger: Logger, input_path: Path = INPUT_CSV):
    from agent import SupportTriageAgent
    retriever = get_retriever()
    agent = SupportTriageAgent(retriever)
    tickets = load_tickets(input_path)

    print(f"[main] Processing {len(tickets)} tickets…\n")
    results = []

    for i, row in enumerate(tickets):
        ticket_id = get_field(row, 'ticket_id', 'id', default=str(i + 1))
        issue     = get_field(row, 'issue', 'ticket_text', 'text', 'description', 'body')
        subject   = get_field(row, 'subject', 'title', default='')
        company   = get_field(row, 'company', 'product', default='')

        result = agent.process(
            issue=issue,
            subject=subject,
            company_field=company,
            ticket_id=ticket_id,
            verbose=True,
        )
        result['ticket_id'] = ticket_id
        logger.log_result(ticket_id, issue, result)
        results.append(result)

    write_output(results, OUTPUT_CSV)
    logger.log(f"\n[main] Done — {len(results)} tickets processed.")
    return results


def do_interactive(logger: Logger):
    from agent import SupportTriageAgent
    retriever = get_retriever()
    agent = SupportTriageAgent(retriever)

    print("\n" + "="*65)
    print("  Support Triage Agent  —  Interactive Mode")
    print("  Enter a support ticket. Blank line to submit. 'quit' to exit.")
    print("="*65)

    n = 0
    while True:
        print()
        lines = []
        print("Issue (blank line to submit, 'quit' to exit):")
        while True:
            line = input("  > ")
            if line.lower() == 'quit':
                print("Goodbye.")
                return
            if line == '':
                break
            lines.append(line)
        if not lines:
            continue

        subject = input("Subject (optional, Enter to skip): ").strip()
        company = input("Company [HackerRank/Claude/Visa/None]: ").strip()

        n += 1
        tid = f"interactive_{n:03d}"
        issue = '\n'.join(lines)

        result = agent.process(
            issue=issue, subject=subject, company_field=company,
            ticket_id=tid, verbose=True
        )
        result['ticket_id'] = tid
        logger.log_result(tid, issue, result)

        print(f"\n{'━'*65}")
        print(f"  STATUS:        {result['status'].upper()}")
        print(f"  PRODUCT AREA:  {result['product_area']}")
        print(f"  REQUEST TYPE:  {result['request_type']}")
        print(f"  JUSTIFICATION: {result['justification']}")
        print(f"\n  RESPONSE:\n{result['response']}")
        print(f"{'━'*65}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Domain Support Triage Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--scrape',       action='store_true', help='Crawl support sites')
    parser.add_argument('--force-scrape', action='store_true', help='Re-crawl even if cached')
    parser.add_argument('--build-index',  action='store_true', help='Build TF-IDF index')
    parser.add_argument('--run',          action='store_true', help='Process support_tickets.csv')
    parser.add_argument('--sample',       action='store_true', help='Run on sample_support_tickets.csv')
    parser.add_argument('--interactive',  action='store_true', help='Interactive mode')
    parser.add_argument('--all',          action='store_true', help='Scrape + index + run')
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ISSUES_DIR.mkdir(parents=True, exist_ok=True)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    logger = Logger(LOG_FILE)

    try:
        if args.all:
            docs = do_scrape(force=args.force_scrape)
            do_build_index(docs)
            do_run(logger)

        elif args.scrape or args.force_scrape:
            do_scrape(force=args.force_scrape)

        elif args.build_index:
            do_build_index()

        elif args.run:
            do_run(logger)

        elif args.sample:
            do_run(logger, input_path=SAMPLE_CSV)

        elif args.interactive:
            do_interactive(logger)

        else:
            print("[main] No command given. Use --help to see options.")
            print("       Defaulting to --interactive mode.\n")
            do_interactive(logger)

    finally:
        logger.close()


if __name__ == '__main__':
    main()
