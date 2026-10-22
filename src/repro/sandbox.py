"""Where agent tools execute.

DockerSandbox is the real one: a throwaway container with no network, no credentials,
no Linux capabilities, a non-root user and CPU / memory / process limits. The repo tree
is copied in (never bind-mounted), so the agent cannot touch the host checkout.

Repos need their dependencies installed, but the sandbox has no network. So each repo
gets an environment image built once *with* network (`ensure_env_image`), and every task
container is started from that image with `--network none`.

LocalSandbox runs commands in a temp dir on the host, with a scrubbed environment but
no isolation and full network access. It exists for tests and for machines without
Docker, and must be opted into explicitly.
"""

import asyncio
import io
import os
import shutil
import sys
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

WORKDIR = "/workspace"
AGENT_UID = 1000

BASE_DOCKERFILE = f"""\
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git ripgrep \\
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytest
RUN useradd -m -u {AGENT_UID} agent && mkdir {WORKDIR} && chown agent {WORKDIR}
WORKDIR {WORKDIR}
"""
BASE_IMAGE = "repro-sandbox-base:latest"

# Best effort: try the common ways Python projects declare test dependencies.
INSTALL_SCRIPT = """\
cd /workspace
pip install --no-cache-dir -e '.[test]' || pip install --no-cache-dir -e '.[tests]' \
  || pip install --no-cache-dir -e '.[dev]' || pip install --no-cache-dir -e . || true
for f in requirements*.txt requirements/*.txt; do
  [ -f "$f" ] && pip install --no-cache-dir -r "$f" || true
done
"""


@dataclass
class ExecResult:
    exit_code: int
    output: str
    timed_out: bool = False


class Sandbox(Protocol):
    async def exec(self, cmd: list[str], timeout: float = 60) -> ExecResult: ...
    async def put_tree(self, tar_bytes: bytes) -> None: ...
    async def write_file(self, rel_path: str, content: str) -> None: ...
    async def close(self) -> None: ...


async def run(
    cmd: list[str], *, stdin: bytes | None = None, timeout: float = 600, env=None, cwd=None
) -> ExecResult:
    """Run a host command, merging stdout and stderr."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        cwd=cwd,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return ExecResult(exit_code=-1, output=f"timed out after {timeout:.0f}s", timed_out=True)
    return ExecResult(exit_code=proc.returncode or 0, output=out.decode(errors="replace"))


def chown_tar(tar_bytes: bytes, uid: int = AGENT_UID) -> bytes:
    """Rewrite every member of a tar to be owned by uid:uid."""
    out = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(tar_bytes)) as src,
        tarfile.open(fileobj=out, mode="w") as dst,
    ):
        for member in src.getmembers():
            member.uid = member.gid = uid
            member.uname = member.gname = ""
            dst.addfile(member, src.extractfile(member) if member.isfile() else None)
    return out.getvalue()


def single_file_tar(rel_path: str, content: str, uid: int = AGENT_UID) -> bytes:
    data = content.encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(rel_path)
        info.size, info.mode, info.uid, info.gid = len(data), 0o644, uid, uid
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class DockerSandbox:
    def __init__(
        self,
        image: str,
        *,
        memory: str = "2g",
        cpus: float = 2.0,
        pids_limit: int = 256,
    ) -> None:
        self.image = image
        self.limits = ["--memory", memory, "--cpus", str(cpus), "--pids-limit", str(pids_limit)]
        self.name = f"repro-{uuid.uuid4().hex[:12]}"
        self.started = False

    async def start(self) -> "DockerSandbox":
        isolation = [
            "--network=none",  # no network at all
            "--cap-drop=ALL",  # no Linux capabilities, even for root
            "--security-opt=no-new-privileges",  # no setuid escalation
            f"--user={AGENT_UID}:{AGENT_UID}",
            f"--workdir={WORKDIR}",
        ]
        # No -e / --env-file: nothing from the host environment reaches the container.
        cmd = ["docker", "run", "-d", "--rm", f"--name={self.name}", *isolation, *self.limits]
        res = await run([*cmd, self.image, "sleep", "infinity"])
        if res.exit_code != 0:
            raise RuntimeError(f"docker run failed: {res.output.strip()}")
        self.started = True
        return self

    async def exec(self, cmd: list[str], timeout: float = 60) -> ExecResult:
        # `timeout` inside the container kills the process tree; wait_for is the backstop.
        inner = ["timeout", "-k", "5", str(int(timeout)), *cmd]
        res = await run(["docker", "exec", self.name, *inner], timeout=timeout + 15)
        if res.exit_code == 124:
            res.timed_out = True
        return res

    async def put_tree(self, tar_bytes: bytes) -> None:
        """Copy a tree into the (empty) workspace, owned by the agent user.

        Ownership is set in the tar and applied by the Docker daemon (`cp -a`), because
        with all capabilities dropped not even root inside the container can chown.
        """
        res = await run(
            ["docker", "cp", "-a", "-", f"{self.name}:{WORKDIR}"], stdin=chown_tar(tar_bytes)
        )
        if res.exit_code != 0:
            raise RuntimeError(f"docker cp failed: {res.output.strip()}")

    async def write_file(self, rel_path: str, content: str) -> None:
        await self.exec(["mkdir", "-p", str(Path(WORKDIR, rel_path).parent)])
        res = await run(
            ["docker", "cp", "-a", "-", f"{self.name}:{WORKDIR}"],
            stdin=single_file_tar(rel_path, content),
        )
        if res.exit_code != 0:
            raise RuntimeError(f"writing {rel_path} failed: {res.output.strip()}")

    async def close(self) -> None:
        if self.started:
            await run(["docker", "rm", "-f", self.name], timeout=60)
            self.started = False


async def image_exists(tag: str) -> bool:
    return (await run(["docker", "image", "inspect", tag], timeout=30)).exit_code == 0


async def ensure_base_image() -> str:
    if not await image_exists(BASE_IMAGE):
        res = await run(
            ["docker", "build", "-t", BASE_IMAGE, "-"], stdin=BASE_DOCKERFILE.encode(), timeout=1800
        )
        if res.exit_code != 0:
            raise RuntimeError(f"building {BASE_IMAGE} failed:\n{res.output[-3000:]}")
    return BASE_IMAGE


async def ensure_env_image(slug: str, tar_bytes: bytes, rebuild: bool = False) -> str:
    """Build (once) an image with this repo's dependencies installed.

    This is the only step with network access. It runs the repo's own install
    hooks, but the container still gets no credentials from the host.
    """
    tag = f"repro-env:{slug.lower()}"
    if not rebuild and await image_exists(tag):
        return tag
    base = await ensure_base_image()
    builder = f"repro-build-{uuid.uuid4().hex[:8]}"
    try:
        res = await run(["docker", "create", "--name", builder, base, "sleep", "infinity"])
        if res.exit_code != 0:
            raise RuntimeError(f"docker create failed: {res.output.strip()}")
        await run(["docker", "start", builder])
        await run(["docker", "cp", "-", f"{builder}:{WORKDIR}"], stdin=tar_bytes)
        install = await run(["docker", "exec", builder, "sh", "-c", INSTALL_SCRIPT], timeout=1800)
        if install.exit_code != 0:
            raise RuntimeError(f"dependency install failed:\n{install.output[-3000:]}")
        # Each task copies its own tree in, so keep only the installed dependencies and
        # leave an empty workspace (stale files from this commit would leak into tasks).
        clear = f"rm -rf {WORKDIR}/* {WORKDIR}/.[!.]* 2>/dev/null; true"
        await run(["docker", "exec", builder, "sh", "-c", clear])
        res = await run(["docker", "commit", builder, tag], timeout=600)
        if res.exit_code != 0:
            raise RuntimeError(f"docker commit failed: {res.output.strip()}")
    finally:
        await run(["docker", "rm", "-f", builder], timeout=60)
    return tag


class LocalSandbox:
    """UNSAFE: no isolation. For tests and machines without Docker only."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="repro-local-"))
        # Scrubbed environment: no API keys or tokens reach the agent's commands.
        self.env = {
            "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
            "HOME": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    async def start(self) -> "LocalSandbox":
        return self

    async def exec(self, cmd: list[str], timeout: float = 60) -> ExecResult:
        if cmd and cmd[0] == "python":
            cmd = [sys.executable, *cmd[1:]]
        return await run(cmd, timeout=timeout, env=self.env, cwd=self.root)

    async def put_tree(self, tar_bytes: bytes) -> None:
        shutil.rmtree(self.root)
        self.root.mkdir()
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            tar.extractall(self.root, filter="data")

    async def write_file(self, rel_path: str, content: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    async def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
