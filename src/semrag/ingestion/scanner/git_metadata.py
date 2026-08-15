"""
Git metadata extraction.

Critical performance decision, stated explicitly because it's the kind of
thing that looks fine in a 50-file test repo and falls over at a million
files: NEVER shell out to `git log` or `git blame` once per file. A
million-file repo would mean a million subprocess spawns, each paying
git's startup cost, for a scan that should complete in minutes.

Instead, this module does exactly two git invocations per scan, regardless
of repo size:

  1. `git ls-tree -r HEAD` — lists every tracked file with its blob SHA in
     one pass. The blob SHA is git's own content hash; if it's unchanged
     between two scans, the file's content is guaranteed unchanged and the
     orchestrator can skip re-hashing (and re-reading) it entirely. This is
     the single biggest incremental-scan perf win in this subsystem.

  2. `git log --name-status --pretty=format:'%H|%an|%aI' <since>..HEAD` —
     one bulk walk of commit history since the last indexed commit, used to
     attribute a `last_commit_sha`/`last_commit_author`/`last_commit_date`
     to every file that changed in that range, without per-file `git log`
     calls. Files untouched since the last scan simply keep whatever
     attribution they already had from the previous manifest.

For repos too large to keep the full commit range in memory in one call
(multi-decade histories), callers should page the `since..HEAD` range —
that pagination is a Part 15 (performance) concern, not implemented here
to keep this module's contract simple: it always operates on one bounded
commit range at a time.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from semrag.ingestion.scanner.models import GitBlobInfo, GitRepositoryMetadata


class GitCommandError(RuntimeError):
    """Raised when a git subprocess exits non-zero. Distinct from a generic
    RuntimeError so the orchestrator can catch it specifically and fall back
    to treating the project as a non-git directory rather than crashing the
    whole scan."""


@dataclass
class _CommitAttribution:
    commit_sha: str
    author: str
    date: datetime


def _run_git(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise GitCommandError(
            f"git {' '.join(args)} failed in {repo_root}: {result.stderr.strip()}"
        )
    return result.stdout


def is_git_repository(path: Path) -> bool:
    try:
        _run_git(path, ["rev-parse", "--is-inside-work-tree"])
        return True
    except (GitCommandError, FileNotFoundError):
        return False


def get_repository_metadata(repo_root: Path) -> GitRepositoryMetadata:
    if not is_git_repository(repo_root):
        return GitRepositoryMetadata(is_git_repo=False)

    branch = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    head_sha = _run_git(repo_root, ["rev-parse", "HEAD"]).strip()

    try:
        remote_url = _run_git(repo_root, ["remote", "get-url", "origin"]).strip()
    except GitCommandError:
        remote_url = None

    status_output = _run_git(repo_root, ["status", "--porcelain"])
    status_lines = [l for l in status_output.splitlines() if l.strip()]
    is_dirty = any(not l.startswith("??") for l in status_lines)
    untracked_count = sum(1 for l in status_lines if l.startswith("??"))

    return GitRepositoryMetadata(
        is_git_repo=True,
        current_branch=branch,
        head_commit_sha=head_sha,
        remote_url=remote_url,
        is_dirty=is_dirty,
        untracked_file_count=untracked_count,
    )


def list_tracked_blobs(repo_root: Path) -> dict[str, str]:
    """One bulk `git ls-tree` call. Returns {relative_path: blob_sha} for
    every file tracked at HEAD. This is the fast-path signal the change
    detector uses to skip re-hashing unchanged files."""
    output = _run_git(repo_root, ["ls-tree", "-r", "-z", "HEAD"])
    blobs: dict[str, str] = {}
    for entry in output.split("\0"):
        if not entry:
            continue
        # format: "<mode> <type> <sha>\t<path>"
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) != 3:
            continue
        _mode, obj_type, sha = parts
        if obj_type == "blob":
            blobs[path] = sha
    return blobs


def bulk_commit_attribution(
    repo_root: Path, since_commit: Optional[str]
) -> dict[str, _CommitAttribution]:
    """One bulk `git log --name-status` call covering the commit range
    (since_commit, HEAD]. Returns {relative_path: attribution} using the
    *most recent* commit touching each path within that range. If
    since_commit is None, walks the full history (first scan of a repo)."""
    commit_range = f"{since_commit}..HEAD" if since_commit else "HEAD"
    output = _run_git(
        repo_root,
        [
            "log",
            "--name-status",
            "--no-renames",
            "--pretty=format:__COMMIT__%H|%an|%aI",
            commit_range,
        ],
    )

    attributions: dict[str, _CommitAttribution] = {}
    current: Optional[_CommitAttribution] = None

    for line in output.splitlines():
        if line.startswith("__COMMIT__"):
            sha, author, date_str = line[len("__COMMIT__"):].split("|", 2)
            current = _CommitAttribution(
                commit_sha=sha, author=author, date=datetime.fromisoformat(date_str)
            )
            continue
        if not line.strip() or current is None:
            continue
        # format: "<status>\t<path>" e.g. "M\tRTL/i2c_master.sv"
        _status, _, path = line.partition("\t")
        path = path.strip()
        # Walking newest-to-oldest: first time we see a path, that's its
        # most recent touch in this range. Don't overwrite with older data.
        if path and path not in attributions:
            attributions[path] = current

    return attributions


def build_blob_info_map(
    repo_root: Path, since_commit: Optional[str]
) -> dict[str, GitBlobInfo]:
    """Combines the two bulk calls above into the per-file GitBlobInfo map
    the orchestrator attaches to each FileRecord."""
    blobs = list_tracked_blobs(repo_root)
    attributions = bulk_commit_attribution(repo_root, since_commit)

    result: dict[str, GitBlobInfo] = {}
    for path, blob_sha in blobs.items():
        attribution = attributions.get(path)
        result[path] = GitBlobInfo(
            blob_sha=blob_sha,
            last_commit_sha=attribution.commit_sha if attribution else None,
            last_commit_author=attribution.author if attribution else None,
            last_commit_date=attribution.date if attribution else None,
            is_tracked=True,
        )
    return result
