import io
import os
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Target repository for test
REPO_URL = "https://github.com/Zoroji/Graph-RAG"
EXTRACT_EXTENSIONS = {".py", ".toml", ".txt", ".json", ".md", ".srt"}
OUTPUT_DIR = Path(__file__).parent / "test_extracted_repo"

def parse_github_url(url: str) -> tuple[str, str]:
    path_parts = urlparse(url).path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: {url}")
    owner, repo = path_parts[0], path_parts[1].removesuffix(".git")
    return owner, repo

def test_zipball_extraction(repo_url: str):
    owner, repo = parse_github_url(repo_url)
    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"
    
    print("=" * 60)
    print(f"🚀 Testing In-Memory Zipball Download & Extraction")
    print(f"📂 Target Repository: {owner}/{repo}")
    print(f"🔗 API Endpoint: {zip_url}")
    print("=" * 60)
    
    # 1. Download using requests (handles cross-domain 302 redirects and TLS negotiation properly)
    t0 = time.perf_counter()
    headers = {
        "User-Agent": "CodeOrbit-Extractor/1.0",
        "Accept": "application/vnd.github+json",
    }
    github_token = os.getenv("GITHUB_TOKEN") or os.getenv("github_token")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
        print("🔑 Using GITHUB_TOKEN from environment.")

    print("\n⏳ Downloading zip archive from GitHub...")
    try:
        import requests
        response = requests.get(zip_url, headers=headers, timeout=(10, 30))
        response.raise_for_status()
        zip_bytes = response.content
    except ImportError:
        print("❌ 'requests' library not found. Run: pip install requests")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n❌ HTTP Error: {e}")
        if response.status_code == 403:
            print("⚠️ Rate limit reached or private repo. Consider setting GITHUB_TOKEN.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Network error: {e}")
        sys.exit(1)

    t_download = time.perf_counter() - t0
    zip_size_kb = len(zip_bytes) / 1024
    print(f"✅ Download completed in {t_download:.2f}s ({zip_size_kb:.1f} KB in-memory)")

    # 2. In-Memory Decompression & Selective Extraction
    t1 = time.perf_counter()
    extracted_files = []
    skipped_files = []

    dest_dir = OUTPUT_DIR / f"{owner}_{repo}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        # GitHub zip archives contain a root wrapper folder: 'owner-repo-commit_sha/'
        namelist = z.namelist()
        root_prefix = namelist[0] if namelist else ""

        for member in z.infolist():
            if member.is_dir():
                continue

            rel_path = member.filename[len(root_prefix):].lstrip("/\\")
            if not rel_path or ".." in rel_path:
                continue

            ext = Path(rel_path).suffix.lower()
            if ext in EXTRACT_EXTENSIONS:
                out_path = dest_dir / rel_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted_files.append((rel_path, member.file_size))
            else:
                skipped_files.append((rel_path, member.file_size))

    t_extract = time.perf_counter() - t1

    # 3. Terminal Report
    print("\n" + "=" * 60)
    print(f"📦 EXTRACTION REPORT: {len(extracted_files)} files saved to '{dest_dir.name}/'")
    print("=" * 60)
    for path, size in extracted_files:
        size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"
        print(f"  📄 [EXTRACTED] {path:<40} ({size_str})")

    if skipped_files:
        print(f"\n🚫 Skipped {len(skipped_files)} non-code / excluded file(s):")
        for path, size in skipped_files[:5]:
            print(f"  ⚪ [SKIPPED]   {path}")
        if len(skipped_files) > 5:
            print(f"  ... and {len(skipped_files) - 5} more")

    print("\n" + "-" * 60)
    print(f"⏱️  Performance Breakdown:")
    print(f"   • Download Time:    {t_download:.2f}s")
    print(f"   • Extraction Time:  {t_extract:.3f}s")
    print(f"   • Total Pipeline:   {t_download + t_extract:.2f}s")
    print("=" * 60)
    print(f"✨ Target folder ready at: {dest_dir.resolve()}\n")

if __name__ == "__main__":
    test_zipball_extraction(REPO_URL)
