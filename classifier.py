"""
classifier.py — Rule-based pre-classifier.

Determines:
  - company         : HackerRank | Claude | Visa | None (inferred)
  - product_area    : fine-grained support category
  - request_type    : product_issue | feature_request | bug | invalid
  - urgency         : low | medium | high | critical
  - should_escalate : bool
  - escalation_reason

All decisions are keyword/regex based — fast, deterministic, no LLM needed here.
The LLM (in agent.py) can override based on corpus retrieval.
"""

import re
from typing import Dict, List, Tuple

# ── Company inference ────────────────────────────────────────────────────────

COMPANY_SIGNALS: Dict[str, List[str]] = {
    "HackerRank": [
        "hackerrank", "hacker rank", "assessment", "coding test", "coding challenge",
        "test environment", "test case", "proctoring", "plagiarism", "candidate invite",
        "recruiter", "ide", "compiler", "submission", "hackerrank pro", "test score",
        "hiring", "skill test", "coding round",
    ],
    "Claude": [
        "claude", "anthropic", "claude.ai", "claude pro", "claude team",
        "claude enterprise", "conversation history", "context window", "artifacts",
        "projects", "claude model", "sonnet", "opus", "haiku", "claude api",
        "claude free", "ai assistant", "chatbot", "llm", "large language model",
    ],
    "Visa": [
        "visa", "visa card", "credit card", "debit card", "transaction", "payment",
        "chargeback", "merchant", "contactless", "chip", "pin", "atm withdrawal",
        "foreign transaction", "visa checkout", "card declined", "authorization",
        "rupee", "upi", "card stolen", "card lost", "visa statement",
    ],
}

# ── Product area mapping ──────────────────────────────────────────────────────

PRODUCT_AREA_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "HackerRank": {
        "IDE & Compiler":           [r"\bide\b", r"compiler", r"code editor", r"run code", r"timeout"],
        "Assessment & Tests":       [r"assessment", r"test", r"coding challenge", r"skill test"],
        "Proctoring & Integrity":   [r"proctoring", r"plagiar", r"cheat", r"camera", r"screen share"],
        "Candidate Experience":     [r"candidate", r"invite", r"link expired", r"retake"],
        "Recruiter & Admin":        [r"recruiter", r"admin", r"dashboard", r"manage"],
        "Account & Access":         [r"login", r"sign.?in", r"password", r"account", r"2fa", r"mfa"],
        "Billing & Plans":          [r"billing", r"plan", r"subscription", r"invoice", r"payment"],
        "Scoring & Results":        [r"score", r"result", r"rank", r"leaderboard"],
        "General":                  [],
    },
    "Claude": {
        "Subscription & Billing":   [r"subscription", r"billing", r"charge", r"plan", r"pro", r"invoice"],
        "Account & Access":         [r"login", r"sign.?in", r"password", r"account", r"verify", r"2fa"],
        "Usage & Limits":           [r"limit", r"message", r"usage", r"quota", r"rate limit"],
        "Features & Artifacts":     [r"artifact", r"feature", r"project", r"memory", r"canvas"],
        "API & Integrations":       [r"\bapi\b", r"sdk", r"integration", r"key", r"token"],
        "Data & Privacy":           [r"data", r"privacy", r"gdpr", r"delete", r"export"],
        "Model & Performance":      [r"model", r"slow", r"response", r"quality", r"hallucin"],
        "General":                  [],
    },
    "Visa": {
        "Card Fraud & Disputes":    [r"fraud", r"unauthorized", r"dispute", r"chargeback", r"stolen"],
        "Lost or Stolen Card":      [r"lost.*card", r"stolen.*card", r"card.*lost", r"card.*stolen", r"misplace"],
        "Transaction Issues":       [r"transaction", r"declined", r"failed.*payment", r"not.*process"],
        "Card Activation":          [r"activat", r"new card", r"register"],
        "ATM & Cash":               [r"\batm\b", r"cash", r"withdraw"],
        "International & Forex":    [r"international", r"foreign", r"forex", r"abroad", r"overseas"],
        "Contactless & Digital":    [r"contactless", r"tap", r"nfc", r"digital wallet", r"upi"],
        "Account & Statements":     [r"statement", r"balance", r"account"],
        "General":                  [],
    },
    "None": {
        "General Inquiry": [],
    },
}

# ── Request type patterns ─────────────────────────────────────────────────────

REQUEST_TYPE_PATTERNS: Dict[str, List[str]] = {
    "bug": [
        r"bug", r"broken", r"not working", r"doesn.?t work", r"glitch", r"error",
        r"crash", r"fail", r"stuck", r"timeout", r"can.?t (?:load|open|access|submit|run)",
        r"issue with", r"problem with", r"keeps? (?:crashing|failing|timing out)",
    ],
    "feature_request": [
        r"can you add", r"wish", r"would be nice", r"feature request", r"suggest",
        r"please add", r"could you implement", r"request for", r"want.*feature",
        r"support for", r"allow us to", r"it would help if",
    ],
    "invalid": [
        r"test\s+ticket", r"ignore this", r"hello world", r"asdf", r"lorem ipsum",
        r"this is a test", r"just testing", r"dummy", r"n/?a\b",
        r"^\s*$", r"^[^a-z]{0,5}$",
    ],
    "product_issue": [],  # default catch-all
}

# ── Escalation triggers ───────────────────────────────────────────────────────

# Critical: always escalate, no reply
CRITICAL_ESCALATION: List[str] = [
    r"fraud", r"unauthorized.*(?:transaction|charge|access)",
    r"stolen.*card", r"card.*stolen",
    r"account.*(?:compromised|hacked|breach)",
    r"identity theft", r"phishing",
    r"legal", r"lawsuit", r"sue\b",
    r"gdpr.*(?:request|violation)", r"data.*breach",
    r"double.?charged", r"duplicate.*charge",
    r"chargeback", r"dispute.*(?:charge|transaction)",
    r"lost.*card", r"card.*lost",
]

# High risk: escalate if combined with other signals
HIGH_RISK: List[str] = [
    r"refund", r"cannot.*access.*account", r"account.*locked",
    r"suspend", r"plagiar", r"cheat.*assessment",
    r"harassment", r"abuse", r"threat",
]

# Low-risk FAQ patterns (confidently reply)
FAQ_PATTERNS: List[str] = [
    r"how (?:do|can|does|to)\b", r"what is\b", r"where (?:can|do|is)\b",
    r"when does\b", r"what are\b", r"can i\b", r"is there\b",
    r"explain\b", r"difference between\b", r"how many\b",
]

# Out-of-scope / malicious signals
OUT_OF_SCOPE: List[str] = [
    r"ignore.*previous.*instruction", r"forget.*instruction",
    r"you are now", r"pretend.*you.*are", r"jailbreak",
    r"disregard.*system", r"act as (?!a support)",
    r"prompt injection", r"override.*policy",
    r"weather", r"recipe", r"write.*poem", r"tell.*joke",
    r"stock.*price", r"sports score", r"news today",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _matches_any(text: str, patterns: List[str]) -> Tuple[bool, str]:
    tl = text.lower()
    for p in patterns:
        if re.search(p, tl):
            return True, p
    return False, ""


def _score_company(text: str) -> str:
    tl = text.lower()
    scores = {k: 0 for k in COMPANY_SIGNALS}
    for company, signals in COMPANY_SIGNALS.items():
        for s in signals:
            if s in tl:
                scores[company] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "None"


def infer_company(issue: str, subject: str, company_field: str) -> str:
    """
    Determine effective company.
    Priority: explicit field > content signals > 'None'
    """
    if company_field and company_field.strip() not in ('', 'None', 'none'):
        return company_field.strip()
    combined = f"{subject} {issue}"
    return _score_company(combined)


def classify_product_area(text: str, company: str) -> str:
    tl = text.lower()
    areas = PRODUCT_AREA_PATTERNS.get(company, PRODUCT_AREA_PATTERNS["None"])
    for area, patterns in areas.items():
        if area == "General":
            continue
        for p in patterns:
            if re.search(p, tl):
                return area
    return list(areas.keys())[-1]  # "General" or last key


def classify_request_type(text: str) -> str:
    tl = text.lower()
    # Check invalid first
    for p in REQUEST_TYPE_PATTERNS["invalid"]:
        if re.search(p, tl):
            return "invalid"
    # Then feature_request
    for p in REQUEST_TYPE_PATTERNS["feature_request"]:
        if re.search(p, tl):
            return "feature_request"
    # Then bug
    for p in REQUEST_TYPE_PATTERNS["bug"]:
        if re.search(p, tl):
            return "bug"
    return "product_issue"


def assess_urgency(text: str, request_type: str) -> str:
    tl = text.lower()
    crit, _ = _matches_any(tl, CRITICAL_ESCALATION)
    if crit:
        return "critical"
    high, _ = _matches_any(tl, HIGH_RISK)
    if high:
        return "high"
    if request_type == "bug":
        return "medium"
    if request_type == "invalid":
        return "low"
    faq, _ = _matches_any(tl, FAQ_PATTERNS)
    return "low" if faq else "medium"


def decide_escalation(text: str, urgency: str, request_type: str) -> Tuple[bool, str]:
    """
    Returns (should_escalate, reason).
    Escalate on:
      - critical urgency (fraud, legal, stolen card, data breach)
      - high urgency security/billing issues
      - invalid/malicious input (out-of-scope injection)
    """
    tl = text.lower()

    # Out-of-scope / injection attempts → escalate with note
    oos, oos_match = _matches_any(tl, OUT_OF_SCOPE)
    if oos and request_type != "feature_request":
        return True, f"Out-of-scope or potentially malicious input (matched: '{oos_match}')"

    # Critical patterns
    crit, crit_match = _matches_any(tl, CRITICAL_ESCALATION)
    if crit:
        return True, f"High-risk issue detected ('{crit_match}') — requires human agent"

    # High-risk patterns
    high, high_match = _matches_any(tl, HIGH_RISK)
    if high and urgency in ("high", "critical"):
        return True, f"Sensitive issue ('{high_match}') — escalating for safety"

    return False, ""


def classify_ticket(issue: str, subject: str = "", company_field: str = "") -> Dict:
    """
    Full ticket classification. Returns a dict with all triage fields.
    """
    combined_text = f"{subject} {issue}".strip()
    company = infer_company(issue, subject, company_field)
    product_area = classify_product_area(combined_text, company)
    request_type = classify_request_type(combined_text)
    urgency = assess_urgency(combined_text, request_type)
    should_escalate, escalation_reason = decide_escalation(combined_text, urgency, request_type)

    return {
        "company": company,
        "product_area": product_area,
        "request_type": request_type,
        "urgency": urgency,
        "should_escalate": should_escalate,
        "escalation_reason": escalation_reason,
    }
