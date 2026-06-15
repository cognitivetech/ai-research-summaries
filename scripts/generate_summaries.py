#!/usr/bin/env python3
"""
Generate paper summaries for cognitivetech/llm-research-summaries repo.

Uses Ollama (qwen3.6 for quality, or glm-4.7-flash for speed).
Outputs to both the cloned repo and a backup directory.
"""

import json
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

# ── Config ─────────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

REPO_BASE = Path.home() / "Sites" / "llm-research-summaries"
BACKUP_DIR = Path.home() / "Sites" / "hermes-lean" / "papers_kb" / "generated_summaries"

# Default model — qwen3.6 for best quality/consistency
MODEL = "qwen3.6:latest"

# Papers to generate: (arxiv_id, target_repo_directory)
# Add more tuples here to batch-generate
PAPERS = [
    # ── Bonus: NeurIPS 2025 paper ──
    ("2605.23464", "training"),  # Unextractable Protocol Models (UPMs)
]

# ── Arxiv Fetching ─────────────────────────────────────────────────────────


def fetch_arxiv_meta(arxiv_id):
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}&max_results=1"
    req = urllib.request.Request(url, headers={"User-Agent": "SummaryBot/1.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    entry = ET.fromstring(resp.read()).find("a:entry", NS)
    if entry is None:
        return None
    title = entry.find("a:title", NS).text.strip().replace("\n", " ").replace("  ", " ")
    abstract = (
        entry.find("a:summary", NS).text.strip().replace("\n", " ").replace("  ", " ")
    )
    authors = [a.find("a:name", NS).text for a in entry.findall("a:author", NS)]
    cats = [c.get("term") for c in entry.findall("a:category", NS)]
    primary = entry.find("arxiv:primary_category", NS)
    if primary is not None:
        primary_term = primary.get("term")
        if primary_term and primary_term not in cats:
            cats.insert(0, primary_term)
    published = entry.find("a:published", NS).text[:10]
    sections = []
    try:
        html_url = f"https://arxiv.org/html/{arxiv_id}"
        html_req = urllib.request.Request(
            html_url, headers={"User-Agent": "SummaryBot/1.0"}
        )
        html_resp = urllib.request.urlopen(html_req, timeout=15)
        html_text = html_resp.read().decode("utf-8", errors="replace")
        headings = re.findall(r"<h[23][^>]*>(.*?)</h[23]>", html_text, re.DOTALL)
        for h in headings[:40]:
            clean = re.sub(r"<[^>]+>", "", h).strip()
            if clean and len(clean) < 200:
                sections.append(clean)
    except Exception:
        pass
    return {
        "id": arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "categories": cats,
        "abstract": abstract,
        "sections": sections,
    }


# ── Ollama Call ────────────────────────────────────────────────────────────


def call_ollama(prompt, model=MODEL):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 16384},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=600)
    result = json.loads(resp.read())
    return result["message"]["content"]


# ── Output Cleaning ────────────────────────────────────────────────────────


def clean_output(content, arxiv_id):
    # Strip code block fences
    content = re.sub(r"^```\w*\n?", "", content)
    content = re.sub(r"\n?```\s*$", "", content)
    # Strip leading YAML frontmatter dashes
    content = re.sub(r"^---\s*\n", "", content)
    content = content.strip()

    # Fix link format: if **Source**: line has bare URL instead of markdown link
    content = re.sub(
        r"\*\*Source\*\*:\s*https?://arxiv\.org/(?:abs|html)/([^\s/]+)",
        r"**Source**: [arXiv:\1](https://arxiv.org/html/\1)",
        content,
    )
    # Fix if **Source**: line has markdown link but wrong URL format
    content = re.sub(
        r"\*\*Source\*\*:\s*\[arXiv:([^\]]+)\]\(https?://arxiv\.org/abs/\1\)",
        r"**Source**: [arXiv:\1](https://arxiv.org/html/\1)",
        content,
    )

    # Ensure arxiv_id appears correctly if it's missing or wrong
    if arxiv_id not in content:
        # Add **Source**: line after title if missing
        if f"https://arxiv.org/abs/{arxiv_id}" not in content:
            content = re.sub(
                r"^(# .+?)\n",
                rf"\1\n\n**Source**: [arXiv:{arxiv_id}](https://arxiv.org/html/{arxiv_id})\n",
                content,
                count=1,
                flags=re.MULTILINE,
            )

    return content.strip()


# ── Prompt Template ────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are writing a structured paper summary for an AI research tracking repository. Generate a detailed section-by-section summary.

## Paper

Title: {title}
Arxiv ID: {arxiv_id}
Authors: {authors}
Published: {published}
Categories: {categories}

Abstract:
{abstract}

Paper sections from HTML (section headings):
{sections_text}

## Output Format

Use EXACTLY this structure:

```
# {title}

**Source**: [arXiv:{arxiv_id}](https://arxiv.org/html/{arxiv_id})

**Authors:** {authors}

## Contents
- [Abstract](#abstract)
- [1 Introduction](#1-introduction)
- [2 Section Name](#2-section-name)
... (all major sections)

## Abstract
**[Key point]**: ...
**Main contribution**: ...
**Key results:**
- ...
**Conclusion:**
- ...

## 1 Introduction
**Context:**
- Point about background
**Goal:**
- What the paper aims to do
**Contributions:**
- ...
```

## Rules
- Use `##` for each major section (Abstract, 1 Introduction, 2 Related Work, 3 Method, 4 Experiments, 5 Results, 6 Discussion, 7 Conclusion, etc.)
- Use `###` for subsections within a major section if the paper has them
- Use bullet points (`- `) throughout — no paragraphs
- Cover every major section from the paper
- Aim for 200-400 lines of bullet-pointed detail
- Use the exact arxiv ID `{arxiv_id}` in the source link
- Start directly with `# Title`, no preamble"""

# ── Generation ─────────────────────────────────────────────────────────────


def slugify(title):
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:120]


def generate_one(arxiv_id, target_dir, model=MODEL, force=False):
    print(f"\n{'=' * 70}")
    print(f"  [{arxiv_id}] → {target_dir}/")

    meta = fetch_arxiv_meta(arxiv_id)
    if not meta:
        print(f"  ❌ Could not fetch arxiv metadata")
        return None

    print(f"  {meta['title'][:70]}")
    print(f"  {', '.join(meta['authors'][:3])}… ({meta['published']})")
    print(f"  Sections: {len(meta['sections'])}")
    sys.stdout.flush()

    # Build prompt
    sections_text = (
        "\n".join(f"  - {s}" for s in meta["sections"])
        if meta["sections"]
        else "  (None extracted)"
    )
    prompt = PROMPT_TEMPLATE.format(
        title=meta["title"],
        arxiv_id=arxiv_id,
        authors=", ".join(meta["authors"]),
        published=meta["published"],
        categories=", ".join(meta["categories"]),
        abstract=meta["abstract"],
        sections_text=sections_text,
    )

    # Call model
    print(f"  Generating ({model.split(':')[0]})… ", end="", flush=True)
    t0 = time.time()
    try:
        raw = call_ollama(prompt, model)
        elapsed = time.time() - t0
        content = clean_output(raw, arxiv_id)
        lines = len(content.split("\n"))
        print(f"{lines} lines in {elapsed:.0f}s")
        sys.stdout.flush()

        # Count sections/subsections
        secs = len(re.findall(r"^##\s+", content, re.MULTILINE))
        subsecs = len(re.findall(r"^###\s+", content, re.MULTILINE))
        bullets = len(re.findall(r"^[\s]*[-*]\s+", content, re.MULTILINE))
        print(f"  Sections: {secs}, Subsections: {subsecs}, Bullets: {bullets}")

        # Validate arxiv ID in output
        if arxiv_id not in content:
            print(f"  ⚠️  arxiv ID not found in generated content — injected")
            # Already handled by clean_output

        # Check if output exists (skip unless --force)
        fname = f"{slugify(meta['title'])}_{arxiv_id}.md"
        repo_path = REPO_BASE / target_dir / fname
        if repo_path.exists() and not force:
            print(f"  ⏭ Skipping, already exists (use --force to overwrite)")
            return repo_path.read_text(encoding="utf-8")

        # Save to repo directory
        repo_path.parent.mkdir(parents=True, exist_ok=True)
        repo_path.write_text(content, encoding="utf-8")
        print(f"  → repo/{target_dir}/{fname}")

        # Save backup
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup = BACKUP_DIR / fname
        backup.write_text(content, encoding="utf-8")
        print(f"  → backup/{fname}")

        return content

    except Exception as e:
        elapsed = time.time() - t0
        print(f"❌ FAILED after {elapsed:.0f}s: {e}")
        sys.stdout.flush()
        return None


def generate_all(papers=None, model=MODEL, force=False):
    papers = papers or PAPERS
    print(f"Generating {len(papers)} summaries")
    print(f"Model: {model}")
    print(f"Repo: {REPO_BASE}")
    print(f"Backup: {BACKUP_DIR}")
    print()

    results = []
    for i, (arxiv_id, target_dir) in enumerate(papers, 1):
        print(f"\n--- [{i}/{len(papers)}] ---")
        content = generate_one(arxiv_id, target_dir, model, force)
        results.append((arxiv_id, content))

    # Summary
    successes = sum(1 for _, c in results if c)
    total_lines = sum(len(c.split("\n")) if c else 0 for _, c in results)
    print(f"\n{'=' * 70}")
    print(f"Done! {successes}/{len(papers)} succeeded, {total_lines} total lines")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper summaries")
    parser.add_argument("--model", default=MODEL, help="Ollama model to use")
    parser.add_argument("--force", action="store_true", help="Regenerate existing")
    parser.add_argument(
        "--repo-base",
        default=str(REPO_BASE),
        help="Path to llm-research-summaries repo (default: ~/Sites/llm-research-summaries)",
    )
    parser.add_argument(
        "--backup-dir",
        default=str(BACKUP_DIR),
        help="Backup directory for generated summaries",
    )
    parser.add_argument(
        "--arxiv",
        help="Generate single paper by arxiv ID + target dir (comma sep: arxiv_id,target_dir)",
    )
    args = parser.parse_args()

    # Apply overrides from CLI
    if args.repo_base:
        global REPO_BASE
        REPO_BASE = Path(args.repo_base)
    if args.backup_dir:
        global BACKUP_DIR
        BACKUP_DIR = Path(args.backup_dir)

    if args.arxiv:
        parts = args.arxiv.split(",")
        arxiv_id = parts[0].strip()
        target_dir = parts[1].strip() if len(parts) > 1 else "models-review"
        generate_one(arxiv_id, target_dir, args.model, args.force)
    else:
        generate_all(model=args.model, force=args.force)
