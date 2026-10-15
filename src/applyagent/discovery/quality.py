"""Explainable shortlist ranking; independent of the discovery platform."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SOFTWARE = re.compile(r"software|\bswe\b|developer|full.?stack|front.?end|back.?end|\bios\b|android|founding.*engineer", re.IGNORECASE)
JUNIOR = re.compile(r"new.?grad|graduate|junior|\bassociate\b|entry.?level|early.?career|\b(engineer|developer) (i|1)\b|\bamts\b", re.IGNORECASE)
SENIOR = re.compile(r"\bsenior\b|\bsr\.?\b|\bstaff\b|\bprincipal\b|\bdirector\b|\bmanager\b|\bintern(ship)?\b", re.IGNORECASE)
NYC = re.compile(r"\bNYC\b|\bNew York City\b|\bNew York,? (?:NY|New York)\b|\bManhattan\b|\bBrooklyn\b|\bQueens\b|\bBronx\b|\bStaten Island\b", re.IGNORECASE)
EXPERIENCE = re.compile(
    r'\b(\d{1,2})\s*(?:(?:[-–—]|to)\s*(\d{1,2}))?\s*\+?\s*years?\s+'
    r'(?:(?:of|as)\s+)?(?:an?\s+)?'
    r'(?:(?:professional|industry|commercial|hands.on|relevant|prior|practical)\s+)*'
    r'(?:experience|software|engineering|programming|development|building|developing|working|backend|frontend|full.stack|ios|android|web|mobile)',
    re.IGNORECASE)


def experience_minima(description: str) -> list[int]:
    """Evidence from experience requirements, excluding optional qualifications."""
    minima = []
    for sentence in re.split(r'[\n.!;]', description):
        if re.search(r'preferred|nice.to.have|bonus|optional|not required', sentence, re.IGNORECASE):
            continue
        minima.extend(int(match[1]) for match in EXPERIENCE.finditer(sentence))
    return minima


def has_early_experience(job: dict) -> bool:
    years = experience_minima(job.get('full_description') or job.get('description') or '')
    return bool(years) and max(years) <= 2


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith("utm_")
             and k.lower() not in {"ref", "source", "referral", "referrer"}]
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/"), urlencode(sorted(query)), ""))


def exclusion_reason(job: dict) -> str | None:
    """Hard exclusions for the current US-based early-career search."""
    title = job.get("title") or ""
    location = job.get("location") or ""
    description = job.get("full_description") or job.get("description") or ""
    if re.search(r"ph\.?d|doctorate", title, re.IGNORECASE):
        return "PhD-track role"
    if re.search(r"customer success|rover|hardware|mechanical|electrical", title, re.IGNORECASE):
        return "Outside the target software-engineering role family"
    foreign = re.search(r"canada|\bcanadian\b|\bUK\b|united kingdom|\bEU\b|europe|india|australia", location, re.IGNORECASE)
    us = re.search(r"\bUS\b|\bUSA\b|united states|worldwide|anywhere|global", location, re.IGNORECASE)
    if foreign and not (us or NYC.search(location)):
        return "Remote geography excludes or does not establish US eligibility"
    for sentence in re.split(r"[!\n]", description):
        if re.search(r"ph\.?d|doctorate", sentence, re.IGNORECASE) and re.search(
            r"is (?:a )?(?:must|required)|must (?:have|hold)|required for this role", sentence, re.IGNORECASE
        ) and not re.search(r"preferred|optional|or equivalent", sentence, re.IGNORECASE):
            return "PhD requirement in description"
    years = experience_minima(description)
    if years and max(years) > 2:
        return 'Minimum experience exceeds the 0–2 year discovery target'
    if re.search(r"\b(?:[5-9]|[1-9][0-9])\+\s*years?\s+(?:of (?:professional |software |engineering )?experience|in applied|of (?:software|engineering))", description, re.IGNORECASE):
        return "Requires at least five years of specialized/professional experience"
    return None


def evaluate(job: dict) -> tuple[int, list[str], list[str]]:
    """Score available evidence, never claim verified eligibility or availability."""
    title = job.get("title") or ""
    location = job.get("location") or ""
    reasons, concerns = [], []
    exclusion = exclusion_reason(job)
    if exclusion:
        return 0, [], [exclusion]
    if not SOFTWARE.search(title) or SENIOR.search(title):
        return 0, [], ["Role/title does not match target level or software work"]
    remote = re.search(r"\bremote\b|anywhere|work from home", location, re.IGNORECASE) and not re.search(
        r"not remote|non.remote|no remote|hybrid", location, re.IGNORECASE)
    nyc = NYC.search(location) and not re.search(r"metropolitan|new york state", location, re.IGNORECASE)
    if not (remote or nyc):
        return 0, [], ["Remote or NYC location not established"]
    score = 30
    reasons.append("Remote" if remote else "NYC location")
    if remote:
        concerns.append("Confirm remote geography and work authorization")
    if re.search(r"founding", title, re.IGNORECASE):
        score += 35
        reasons.append("Founding engineering role")
        concerns.append("Confirm required experience for founding role")
    elif JUNIOR.search(title):
        score += 35
        reasons.append("Explicit early-career title")
    elif has_early_experience(job):
        score += 35
        reasons.append('Description asks for 0–2 years of experience')
        concerns.append('Verify exact professional/specialized experience and graduation/start requirements')
    elif job.get("strategy") == "github_repo":
        score += 25
        reasons.append("Listed in a curated new-grad repository")
    else:
        score += 20
        concerns.append('General SWE title: verify 1–2 year suitability from the employer requirements')
    if job.get("company"):
        score += 5
        reasons.append("Named employer")
    description = job.get("full_description") or job.get("description") or ""
    if len(description) > 200:
        score += 5
        reasons.append("Description available")
    else:
        concerns.append("Full description needs review")
    concerns.append("Posting availability and eligibility not verified")
    return score, reasons, concerns


def rank_jobs(jobs: list[dict], limit: int = 25, min_score: int = 0, max_per_company: int = 3) -> list[dict]:
    if max_per_company < 1:
        raise ValueError('max_per_company must be positive')
    ranked = []
    for job in jobs:
        score, reasons, concerns = evaluate(job)
        if score and score >= min_score:
            ranked.append({**job, "match_score": score, "match_reasons": reasons, "review_needed": concerns})
    ranked.sort(key=lambda j: (not (bool(JUNIOR.search(j.get('title') or '')) or has_early_experience(j)),
                               bool(re.search('founding', j.get('title') or '', re.IGNORECASE)),
                               -j["match_score"], j.get("title") or "", j["url"]))
    result, seen, companies = [], set(), {}
    for job in ranked:
        target = job.get("application_url") or job["url"]
        key = canonical_url(target)
        if key in seen:
            continue
        company = (job.get('company') or '').casefold().strip()
        if company and companies.get(company, 0) >= max_per_company:
            continue
        companies[company] = companies.get(company, 0) + 1
        seen.add(key)
        # Keep reports compact; descriptions remain available in SQLite.
        job.pop("description", None)
        job.pop("full_description", None)
        result.append(job)
    return result[:limit]
