import io
import os
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import requests

# Code and configuration file extensions we want to retain
SUPPORTED_EXTENSIONS = {
    # Python
    ".py", ".pyi",
    # JavaScript & TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Systems & Compiled
    ".go", ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".kt", ".swift",
    # Scripting & Web
    ".rb", ".php", ".sh", ".bash", ".sql", ".html", ".css", ".scss",
    # Config & Manifests
    ".json", ".yaml", ".yml", ".toml", ".txt", ".md"
}

# Directories to skip entirely
EXCLUDED_DIRS = {
    ".git", ".github", "node_modules", "venv", ".venv", "env",
    "__pycache__", "dist", "build", ".next", ".nuxt", "coverage", "vendor"
}


def parse_github_url(url: str) -> tuple[str, str]:
    """Extract owner and repo name from a public GitHub URL."""
    path_parts = urlparse(url).path.strip("/").split("/")
    if len(path_parts) < 2:
        raise ValueError(f"Invalid GitHub URL: '{url}'. Expected format: https://github.com/owner/repo")
    owner, repo = path_parts[0], path_parts[1].removesuffix(".git")
    return owner, repo


def extract_github_zipball(
    repo_url: str,
    target_base_dir: Path,
    github_token: str | None = None
) -> tuple[Path, str, str, bool]:
    """
    Downloads repository zipball directly from GitHub and selectively extracts code files.
    Returns: (repo_dir, owner, repo, was_cached)
    """
    owner, repo = parse_github_url(repo_url)
    repo_id = f"{owner}_{repo}"
    dest_dir = target_base_dir / repo_id

    # 1. Zero-latency cache hit
    if dest_dir.exists() and any(dest_dir.iterdir()):
        return dest_dir, owner, repo, True

    # 2. Prepare request
    headers = {
        "User-Agent": "CodeOrbit-Engine/1.0",
        "Accept": "application/vnd.github+json",
    }
    token = github_token or os.getenv("GITHUB_TOKEN") or os.getenv("github_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    zip_url = f"https://api.github.com/repos/{owner}/{repo}/zipball"

    # 3. Stream download into memory
    response = requests.get(zip_url, headers=headers, stream=True, timeout=(10, 30))
    response.raise_for_status()

    # 4. In-memory selective extraction
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        namelist = z.namelist()
        root_prefix = namelist[0] if namelist else ""

        for member in z.infolist():
            if member.is_dir():
                continue

            # Strip GitHub root folder wrapper
            rel_path = member.filename[len(root_prefix):].lstrip("/\\")
            if not rel_path or ".." in rel_path:
                continue

            path_obj = Path(rel_path)

            # Skip ignored directories (e.g. node_modules, .git, venv)
            if any(part in EXCLUDED_DIRS for part in path_obj.parts):
                continue

            # Filter only code and manifest extensions
            if path_obj.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            # Prevent extracting massive binaries or minified bundles (> 2MB)
            if member.file_size > 2 * 1024 * 1024:
                continue

            out_path = dest_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())

    return dest_dir, owner, repo, False
