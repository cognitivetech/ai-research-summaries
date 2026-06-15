#!/usr/bin/env python3
"""
Build a unified AI/ML research paper knowledge base from five sources:
1. The AI Timeline (@TheAITimeline) - ~1,500 papers (local index file)
2. DAIR.AI (@dair_ai) - ~1,775 papers (local index file)
3. cognitivetech/llm-research-summaries - 135 paper summaries (fetched from GitHub)
4. NeurIPS 2025 proceedings - 5,800+ conference papers
5. cognitivetech/arxiv-daily - 10,000+ papers (fetched, filtered to relevant categories)

Output: JSONL (one JSON object per line) + summary JSON index
"""

import json
import os
import re
import sys
import urllib.request
import hashlib
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "papers_kb"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_JSONL = OUTPUT_DIR / "papers.jsonl"
OUTPUT_INDEX = OUTPUT_DIR / "index.json"
OUTPUT_STATS = OUTPUT_DIR / "stats.json"

TIMELINE_FILE = PROJECT_ROOT / "the_ai_timeline_papers_index.md"
DAIR_FILE = PROJECT_ROOT / "dair_ai_papers_index.md"

CACHE_DIR = PROJECT_ROOT / ".kb_cache"
CACHE_DIR.mkdir(exist_ok=True)

# arxiv-daily categories relevant to AI/ML/LLM
ARXIV_DAILY_CATEGORIES = {
    "Reinforcement Learning",
    "Transformer",
    "Multi-modal",
    "Graph Neural Network",
    "Robotics",
    "Image Classification",
    "Object Detection",
    "Semantic Segmentation",
    "Instance Segmentation",
    "Object Tracking",
    "Multi-Object Tracking",
    "Keypoint Detection",
    "Image Matching",
    "3D Object Detection",
    "3D Object Tracking",
    "Point Cloud",
    "Point Cloud Segmentation",
    "Federated Learning",
    "Few-shot Learning",
    "Unsupervised Learning",
    "Transfer Learning",
    "Contrastive Learning",
}

# ── Helpers ────────────────────────────────────────────────────────────────


def paper_id(title, source):
    return hashlib.md5(f"{source}::{title}".encode()).hexdigest()[:12]


def fetch_url(url, timeout=30):
    """Fetch URL with disk caching to avoid rate limits."""
    cache_path = CACHE_DIR / hashlib.md5(url.encode()).hexdigest()[:16]
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    headers = {"User-Agent": "Hermes-KB-Builder/1.0"}
    gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if gh_token and "github.com" in url:
        headers["Authorization"] = f"Bearer {gh_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read().decode("utf-8")
    cache_path.write_text(data, encoding="utf-8")
    return data


def normalize_title(title):
    t = re.sub(r"\([^)]*\)", "", title.strip())
    t = re.sub(r"\[[^\]]*\]", "", t)
    t = re.sub(r"\*\*", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


# ── Source 1: The AI Timeline ──────────────────────────────────────────────


def parse_timeline(filepath):
    papers = []
    text = filepath.read_text(encoding="utf-8")
    seen_titles = set()

    # Extract thread papers
    for m in re.finditer(
        r"### Thread \d+.*?\n(.*?)(?=\n### |\n---|\Z)", text, re.DOTALL
    ):
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("-"):
                continue
            line = line.lstrip("- ").strip()
            arxiv_match = re.search(r"(\d{4}\.\d{4,5})", line)
            title = re.sub(
                r"\s*\[\d{4}\.\d{4,5}\]\(https?://arxiv.org/abs/\d{4}\.\d{4,5}\)",
                "",
                line,
            )
            title = re.sub(r"\s*\(\d{4}\.\d{4,5}\)", "", title)
            title = re.sub(r"\s*\(\)", "", title).strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            papers.append({
                "id": paper_id(title, "the_ai_timeline"),
                "title": title,
                "source": "the_ai_timeline",
                "source_detail": "thread",
                "category": [],
                "source_categories": [],
                "arxiv_id": arxiv_match.group(1) if arxiv_match else None,
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_match.group(1)}"
                if arxiv_match
                else None,
                "date": None,
                "authors": [],
                "summary": None,
                "abstract": None,
            })

    # Extract newsletter alphabetical index
    m = re.search(r"## 📰 Newsletter Issues.*?\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if m:
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("- "):
                continue
            title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line[2:]).strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            papers.append({
                "id": paper_id(title, "the_ai_timeline"),
                "title": title,
                "source": "the_ai_timeline",
                "source_detail": "newsletter",
                "category": [],
                "source_categories": [],
                "arxiv_id": None,
                "arxiv_url": None,
                "date": None,
                "authors": [],
                "summary": None,
                "abstract": None,
            })
    return papers


# ── Source 2: DAIR.AI ─────────────────────────────────────────────────────

DAIR_CATEGORY_MAP = {
    "agents": "agents",
    "reasoning": "reasoning",
    "architecture": "architecture",
    "training": "training",
    "inference": "inference",
    "safety": "safety",
    "rag": "rag",
    "multimodal": "multimodal",
    "evaluation": "evaluation",
    "scientific": "scientific/domain",
    "code": "code/se",
    "speech": "speech/audio",
    "model releases": "model_releases",
}


def parse_dair(filepath):
    papers = []
    text = filepath.read_text(encoding="utf-8")
    seen_titles = set()

    for m in re.finditer(r"### .+?\(~(\d+).*?\)\n(.*?)(?=\n### |\Z)", text, re.DOTALL):
        header = m.group(0).split("\n")[0]
        content = m.group(2)
        section_lower = header.lower()
        category = "uncategorized"
        source_category = header.replace("### ", "").strip()
        for key, val in DAIR_CATEGORY_MAP.items():
            if key in section_lower:
                category = val
                break

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or not line.startswith("- "):
                continue
            title = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line[2:]).strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            papers.append({
                "id": paper_id(title, "dair_ai"),
                "title": title,
                "source": "dair_ai",
                "source_detail": source_category,
                "category": [category] if category != "uncategorized" else [],
                "source_categories": [source_category],
                "arxiv_id": None,
                "arxiv_url": None,
                "date": None,
                "authors": [],
                "summary": None,
                "abstract": None,
            })
    return papers


# ── Source 3: llm-research-summaries ──────────────────────────────────────

SUMMARY_CAT_MAP = {
    "models-review": "architecture",
    "prompting": "reasoning",
    "training": "training",
    "steering": "safety",
    "rag": "rag",
    "agents": "agents",
    "inference": "inference",
    "multimodal": "multimodal",
    "evaluation": "evaluation",
    "reasoning": "reasoning",
    "safety": "safety",
    "optimization": "optimizations",
}


def fetch_llm_summaries():
    papers = []
    seen_titles = set()
    base = "https://api.github.com/repos/cognitivetech/llm-research-summaries/contents"

    try:
        items = json.loads(fetch_url(base))
    except Exception as e:
        print(
            f"  [WARN] llm-research-summaries root fetch failed: {e}", file=sys.stderr
        )
        return papers

    for item in items:
        if item["type"] != "dir":
            continue
        dir_name = item["name"]
        cat = SUMMARY_CAT_MAP.get(dir_name, dir_name)
        try:
            dir_items = json.loads(fetch_url(item["url"]))
        except Exception:
            continue
        for fi in dir_items:
            if not fi["name"].endswith((".md", ".mdx")):
                continue
            if fi["name"] in ("README.md", "readme.md", "index.md"):
                continue
            stem = fi["name"].rsplit(".", 1)[0]
            title = re.sub(r"[-_]", " ", stem).title()
            title = re.sub(r"\s+", " ", title).strip()
            if not title or title.lower() in seen_titles:
                continue
            seen_titles.add(title.lower())
            summary = None
            try:
                content = fetch_url(fi["download_url"])
                content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)
                content = re.sub(r"#{1,6}\s+", "", content)
                paragraphs = [
                    p.strip() for p in content.split("\n\n") if len(p.strip()) > 50
                ]
                if paragraphs:
                    summary = paragraphs[0][:1000]
            except Exception:
                pass
            papers.append({
                "id": paper_id(title, "llm_research_summaries"),
                "title": title,
                "source": "llm_research_summaries",
                "source_detail": dir_name,
                "category": [cat],
                "source_categories": [dir_name],
                "arxiv_id": None,
                "arxiv_url": None,
                "date": None,
                "authors": [],
                "summary": summary,
                "abstract": None,
            })
    return papers


# ── Source 4: NeurIPS 2025 Proceedings ─────────────────────────────────────

NEURIPS_2025_URL = "https://proceedings.neurips.cc/paper_files/paper/2025"


def fetch_neurips():
    papers = []
    seen_titles = set()
    base_url = NEURIPS_2025_URL

    try:
        html = fetch_url(base_url)
    except Exception as e:
        print(f"  [WARN] NeurIPS fetch failed: {e}", file=sys.stderr)
        return papers

    # Parse paper cards from the HTML list
    for match in re.finditer(
        r'<li\s+class="([^"]*)"[^>]*data-track="([^"]*)">\s*'
        r'<div class="paper-content">\s*'
        r'<a[^>]*href="([^"]*)"[^>]*>([^<]+)</a>\s*'
        r'<span class="paper-authors">([^<]*)</span>',
        html,
        re.DOTALL,
    ):
        track_class = match.group(1)
        track_data = match.group(2)
        href = match.group(3)
        title = match.group(4).strip()
        authors_text = match.group(5).strip()

        # Normalize title
        title = re.sub(r"\s+", " ", title).strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())

        # Parse authors
        authors = [
            a.strip().rstrip(".")
            for a in re.split(r",\s*(?=[A-Z])", authors_text)
            if a.strip()
        ]
        if not authors:
            authors = []

        # Determine track name
        track_map = {
            "conference": "Main Conference Track",
            "position_paper_track": "Position Paper Track",
            "datasets_and_benchmarks_track": "Datasets and Benchmarks Track",
        }
        track_name = track_map.get(track_data, track_data)

        # Build full URL
        full_url = (
            f"https://proceedings.neurips.cc{href}" if href.startswith("/") else href
        )

        papers.append({
            "id": paper_id(title, "neurips"),
            "title": title,
            "source": "neurips_2025",
            "source_detail": track_name,
            "category": [],
            "source_categories": [track_name],
            "arxiv_id": None,
            "arxiv_url": None,
            "date": "2025-12",
            "authors": authors,
            "summary": None,
            "abstract": None,
            "neurips_url": full_url,
        })

    print(f"    Parsed {len(papers)} papers from proceedings page")
    return papers


# ── Source 5: arxiv-daily ─────────────────────────────────────────────────

ARXIV_CAT_MAP = {
    "Reinforcement Learning": "training",
    "Transformer": "architecture",
    "Multi-modal": "multimodal",
    "Graph Neural Network": "architecture",
    "Robotics": "agents",
    "Image Classification": "multimodal",
    "Object Detection": "multimodal",
    "Semantic Segmentation": "multimodal",
    "Instance Segmentation": "multimodal",
    "Object Tracking": "multimodal",
    "Multi-Object Tracking": "multimodal",
    "Keypoint Detection": "multimodal",
    "Image Matching": "multimodal",
    "3D Object Detection": "multimodal",
    "3D Object Tracking": "multimodal",
    "Point Cloud": "multimodal",
    "Point Cloud Segmentation": "multimodal",
    "Federated Learning": "training",
    "Few-shot Learning": "training",
    "Unsupervised Learning": "training",
    "Transfer Learning": "training",
    "Contrastive Learning": "training",
}


def fetch_arxiv_daily():
    papers = []
    seen_titles = set()
    url = "https://raw.githubusercontent.com/cognitivetech/arxiv-daily/main/README.md"
    try:
        text = fetch_url(url)
    except Exception as e:
        print(f"  [WARN] arxiv-daily fetch failed: {e}", file=sys.stderr)
        return papers

    for section in re.split(r"\n## ", text):
        if not section.strip():
            continue
        cat_name = section.split("\n")[0].strip()
        if cat_name not in ARXIV_DAILY_CATEGORIES:
            continue

        for sub in re.split(r"\n### ", section)[1:]:
            lines = sub.strip().split("\n")
            subcat = lines[0].strip()
            in_table = False
            for line in lines:
                line = line.strip()
                if not line.startswith("|"):
                    in_table = False
                    continue
                if "---" in line and "|" in line:
                    in_table = True
                    continue
                if not in_table:
                    continue
                if line.count("|") < 5:
                    continue
                cols = [c.strip() for c in line.split("|")]
                title = cols[2] if len(cols) > 2 else ""
                authors_col = cols[3] if len(cols) > 3 else ""
                pdf_link = cols[4] if len(cols) > 4 else ""
                if not title:
                    continue
                title_clean = re.sub(r"\*\*", "", title).strip()
                arxiv_match = re.search(r"(\d{4}\.\d{4,5})", pdf_link)
                arxiv_id = arxiv_match.group(1) if arxiv_match else None
                if arxiv_id:
                    title_clean = re.sub(
                        r"\s*\[\d{4}\.\d{4,5}v?\d?\]\([^)]+\)", "", title_clean
                    )
                    title_clean = title_clean.strip()
                if not title_clean or title_clean.lower() in seen_titles:
                    continue
                seen_titles.add(title_clean.lower())
                authors = []
                if authors_col:
                    for a in re.split(r",\s*", authors_col):
                        a = a.strip().lstrip("*").rstrip("*")
                        if a and a != "et.al.":
                            authors.append(a.rstrip(".,"))
                papers.append({
                    "id": paper_id(title_clean, "arxiv_daily"),
                    "title": title_clean,
                    "source": "arxiv_daily",
                    "source_detail": f"{cat_name} / {subcat}",
                    "category": [ARXIV_CAT_MAP.get(cat_name, "uncategorized")],
                    "source_categories": [cat_name, subcat],
                    "arxiv_id": arxiv_id,
                    "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}"
                    if arxiv_id
                    else None,
                    "date": re.sub(r"\*\*", "", cols[1]).strip()
                    if len(cols) > 1
                    else None,
                    "authors": authors,
                    "summary": None,
                    "abstract": None,
                })
    return papers


# ── Auto-Categorization ───────────────────────────────────────────────────

CAT_KEYWORDS = {
    "agents": [
        "agent",
        "multi-agent",
        "autonomous",
        "tool use",
        "orchestrat",
        "delegat",
        "react",
        "function calling",
        "tool-call",
    ],
    "reasoning": [
        "reasoning",
        "chain-of-thought",
        "cot",
        "thought",
        "think",
        "test-time",
        "inference-time",
        "deep think",
        "logic",
        "proof",
        "theorem",
        "r1",
        "o1",
    ],
    "architecture": [
        "transformer",
        "attention",
        "mamba",
        "state space",
        "ssm",
        "moe",
        "mixture of expert",
        "rnn",
        "lstm",
        "tokenizer",
        "embedding",
        "positional",
        "normalization",
        "activation",
        "gating",
        "softmax",
    ],
    "training": [
        "training",
        "pretraining",
        "pre-training",
        "finetuning",
        "fine-tuning",
        "rlhf",
        "dpo",
        "alignment",
        "sft",
        "reinforcement learning",
        "reinforce",
        "optimization",
        "gradient",
        "scaling law",
        "data mix",
        "synthetic data",
    ],
    "inference": [
        "inference",
        "kv cache",
        "quantization",
        "pruning",
        "distillation",
        "speculative decoding",
        "compression",
        "flash attention",
        "efficient",
    ],
    "safety": [
        "safety",
        "alignment",
        "jailbreak",
        "hallucination",
        "watermark",
        "bias",
        "fairness",
        "unlearning",
        "red team",
        "adversarial",
        "misalignment",
        "toxic",
        "harm",
    ],
    "rag": ["rag", "retrieval-augmented", "retrieval", "knowledge base"],
    "multimodal": [
        "multimodal",
        "vision",
        "image",
        "video",
        "vlm",
        "visual",
        "captioning",
        "text-to-image",
        "diffusion",
        "generation",
        "3d",
        "nerf",
        "segmentation",
        "object detection",
    ],
    "evaluation": [
        "benchmark",
        "evaluation",
        "eval",
        "leaderboard",
        "metric",
        "dataset",
        "test",
    ],
    "scientific/domain": [
        "scientific",
        "biology",
        "chemistry",
        "physics",
        "medical",
        "health",
        "drug",
        "protein",
        "genome",
        "climate",
        "mathematical",
    ],
    "code/se": [
        "code",
        "software",
        "programming",
        "swe-bench",
        "debug",
        "compiler",
        "repository",
    ],
    "speech/audio": ["speech", "audio", "voice", "tts", "asr", "music", "sound"],
}


def auto_categorize(title):
    title_lower = title.lower()
    cats = set()
    for cat, keywords in CAT_KEYWORDS.items():
        for kw in keywords:
            if kw in title_lower:
                cats.add(cat)
                break
    return sorted(cats)


# ── Deduplication ─────────────────────────────────────────────────────────


def deduplicate(papers):
    title_map = {}
    for p in papers:
        key = normalize_title(p["title"])
        if not key:
            continue
        if key in title_map:
            e = title_map[key]
            if p.get("arxiv_id") and not e.get("arxiv_id"):
                e["arxiv_id"] = p["arxiv_id"]
                e["arxiv_url"] = p["arxiv_url"]
            if p.get("summary") and not e.get("summary"):
                e["summary"] = p["summary"]
            if p.get("abstract") and not e.get("abstract"):
                e["abstract"] = p["abstract"]
            if p.get("date") and not e.get("date"):
                e["date"] = p["date"]
            if p.get("authors") and not e.get("authors"):
                e["authors"] = p["authors"]
            e["_sources"].append(p["source"])
            e["category"] = list(set(e.get("category", []) + p.get("category", [])))
            e["source_categories"] = list(
                set(e.get("source_categories", []) + p.get("source_categories", []))
            )
        else:
            p["_sources"] = [p["source"]]
            title_map[key] = p

    result = []
    for p in title_map.values():
        p["sources"] = sorted(set(p["_sources"]))
        if len(p["sources"]) == 1:
            p["source"] = p["sources"][0]
        else:
            p["source"] = "+".join(p["sources"])
        del p["_sources"]
        del p["source_detail"]
        result.append(p)
    return result


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    print("Building AI/ML Research Paper Knowledge Base")
    print("=" * 60)
    all_papers = []

    # Source 1
    print("\n[1/4] Parsing The AI Timeline...")
    if TIMELINE_FILE.exists():
        n = len(all_papers)
        all_papers.extend(parse_timeline(TIMELINE_FILE))
        print(f"  +{len(all_papers) - n} papers")

    # Source 2
    print("\n[2/4] Parsing DAIR.AI...")
    if DAIR_FILE.exists():
        n = len(all_papers)
        all_papers.extend(parse_dair(DAIR_FILE))
        print(f"  +{len(all_papers) - n} papers")

    # Source 3
    print("\n[3/5] Fetching llm-research-summaries...")
    n = len(all_papers)
    all_papers.extend(fetch_llm_summaries())
    print(f"  +{len(all_papers) - n} papers")

    # Source 4
    print("\n[4/5] Fetching NeurIPS 2025 proceedings...")
    n = len(all_papers)
    all_papers.extend(fetch_neurips())
    print(f"  +{len(all_papers) - n} papers")

    # Source 5
    print("\n[5/5] Fetching arxiv-daily...")
    n = len(all_papers)
    all_papers.extend(fetch_arxiv_daily())
    print(f"  +{len(all_papers) - n} papers")

    # Auto-categorize
    uncat = sum(1 for p in all_papers if not p.get("category"))
    if uncat:
        print(f"\nAuto-categorizing {uncat} uncategorized papers...")
        for p in all_papers:
            if not p.get("category"):
                p["category"] = auto_categorize(p["title"])
        print(f"  Done")

    # Deduplicate
    print(f"\n{'=' * 60}")
    print(f"Raw total: {len(all_papers)} papers")
    print("Deduplicating...")
    deduped = deduplicate(all_papers)
    print(f"Unique:    {len(deduped)} papers")

    # Write JSONL
    print(f"\nWriting {OUTPUT_JSONL}...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for p in deduped:
            out = {k: v for k, v in p.items() if v is not None}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    # Write index
    print(f"Writing {OUTPUT_INDEX}...")
    by_title = {}
    by_arxiv = {}
    for p in deduped:
        key = normalize_title(p["title"])
        entry = {
            "id": p["id"],
            "title": p["title"],
            "sources": p.get("sources", [p["source"]]),
            "arxiv_url": p.get("arxiv_url"),
            "category": p.get("category", []),
        }
        by_title[key] = entry
        if p.get("arxiv_id"):
            by_arxiv[p["arxiv_id"]] = {"id": p["id"], "title": p["title"]}
    with open(OUTPUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(
            {"by_title": by_title, "by_arxiv_id": by_arxiv},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Write stats
    print(f"Writing {OUTPUT_STATS}...")
    from collections import Counter

    src_counts = Counter()
    cat_counts = Counter()
    for p in deduped:
        for s in p.get("sources") or [p["source"]]:
            src_counts[s] += 1
        for c in p.get("category", []):
            cat_counts[c] += 1

    stats = {
        "total_papers": len(deduped),
        "deduped_from": len(all_papers),
        "by_source": dict(src_counts.most_common()),
        "by_category": dict(cat_counts.most_common()),
        "papers_with_arxiv": sum(1 for p in deduped if p.get("arxiv_id")),
        "papers_with_authors": sum(1 for p in deduped if p.get("authors")),
        "papers_with_summary": sum(1 for p in deduped if p.get("summary")),
    }
    with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Knowledge base: {OUTPUT_DIR}")
    print(f"  papers.jsonl  — {len(deduped)} papers")
    print(f"  index.json    — title + arxiv_id lookup")
    print(f"  stats.json    — summary stats")
    print(
        f"  {stats['papers_with_arxiv']} with arxiv links, "
        f"{stats['papers_with_summary']} with summaries"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build AI/ML research paper knowledge base"
    )
    parser.add_argument(
        "--reset-cache", action="store_true", help="Clear cached API responses"
    )
    args = parser.parse_args()
    if args.reset_cache:
        import shutil

        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        print("Cache cleared.")
    main()
