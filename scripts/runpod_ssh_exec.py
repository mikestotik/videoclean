#!/usr/bin/env python3
"""Run commands / upload a file on a RunPod via TCP SSH or ssh.runpod.io proxy.

Used by GitHub Actions. Proxy path allocates a PTY (required by RunPod basic SSH).
"""
from __future__ import annotations

import argparse
import base64
import os
import pty
import re
import select
import subprocess
import sys
import time
from typing import Optional


def _pump(fd: int, buf: list[str], seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.4)
        if fd not in r:
            continue
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            return
        if not chunk:
            return
        text = chunk.decode(errors="replace")
        buf.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()


def _wait_prompt(fd: int, buf: list[str], timeout: float = 45) -> None:
    end = time.time() + timeout
    while time.time() < end:
        _pump(fd, buf, 0.8)
        tail = "".join(buf)[-400:]
        if re.search(r"root@[^#\n]*#\s*$", tail):
            return
    raise RuntimeError("timed out waiting for shell prompt")


def _write(fd: int, data: str) -> None:
    os.write(fd, data.encode())


def ssh_session(
    ssh_argv: list[str],
    remote_script: str,
    upload: Optional[tuple[str, str]] = None,
    timeout: float = 900,
) -> str:
    """Open interactive SSH, optional upload local_path -> remote_path, run script."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(ssh_argv[0], ssh_argv)

    buf: list[str] = []
    marker = f"__RP_DONE_{int(time.time())}__"
    try:
        _wait_prompt(fd, buf)
        if upload:
            local_path, remote_path = upload
            with open(local_path, "rb") as f:
                raw = f.read()
            b64 = base64.b64encode(raw).decode()
            remote_b64 = "/tmp/_rp_upload.b64"
            _write(fd, f"rm -f {remote_b64} {remote_path}\n")
            _pump(fd, buf, 1.5)
            _write(fd, f"cat > {remote_b64} <<'B64EOF'\n")
            _pump(fd, buf, 0.8)
            step = 400
            for i in range(0, len(b64), step):
                _write(fd, b64[i : i + step] + "\n")
                if i % 20000 == 0:
                    _pump(fd, buf, 0.05)
            _write(fd, "B64EOF\n")
            _pump(fd, buf, 2)
            _write(
                fd,
                f"base64 -d {remote_b64} > {remote_path} && "
                f"wc -c {remote_path} && rm -f {remote_b64}\n",
            )
            _pump(fd, buf, 5)

        wrapped = remote_script.rstrip() + f"\necho {marker}\n"
        script_b64 = base64.b64encode(wrapped.encode()).decode()
        _write(fd, "cat > /tmp/_rp_cmd.b64 <<'B64EOF'\n")
        _pump(fd, buf, 0.5)
        step = 400
        for i in range(0, len(script_b64), step):
            _write(fd, script_b64[i : i + step] + "\n")
        _write(fd, "B64EOF\n")
        _pump(fd, buf, 1)
        _write(fd, "base64 -d /tmp/_rp_cmd.b64 > /tmp/_rp_cmd.sh && bash /tmp/_rp_cmd.sh\n")

        end = time.time() + timeout
        out = ""
        while time.time() < end:
            _pump(fd, buf, 3)
            out = "".join(buf)
            if marker in out:
                break
        else:
            raise RuntimeError("remote command timed out")

        _write(fd, "exit\n")
        _pump(fd, buf, 2)
        return "".join(buf)
    finally:
        try:
            os.kill(pid, 9)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


def resolve_tcp(pod_json: str) -> Optional[list[str]]:
    import json

    pod = json.loads(pod_json)
    ip = pod.get("publicIp") or ""
    mappings = pod.get("portMappings") or {}
    port = mappings.get("22") or mappings.get(22)
    if not ip or not port:
        return None
    key = os.environ["RUNPOD_SSH_KEY_PATH"]
    return [
        "ssh",
        "-tt",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=15",
        "-i",
        key,
        "-p",
        str(port),
        f"root@{ip}",
    ]


def fetch_proxy_user_from_api(pod_id: str) -> str:
    """Resolve ssh.runpod.io username via REST v2 (no manual suffix needed)."""
    import json
    import urllib.error
    import urllib.request

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        return ""
    req = urllib.request.Request(
        f"https://api.runpod.io/v2/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            pod = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"v2 pod lookup failed: {exc}", file=sys.stderr)
        return ""
    ssh = pod.get("ssh") or {}
    proxy = ssh.get("proxy") or {}
    user = (proxy.get("username") or "").strip()
    if user:
        print(f"resolved proxy user from API: {user}", file=sys.stderr)
    return user


def resolve_proxy(pod_id: str) -> list[str]:
    key = os.environ["RUNPOD_SSH_KEY_PATH"]
    user = os.environ.get("RUNPOD_SSH_PROXY_USER", "").strip()
    if not user:
        suffix = os.environ.get("RUNPOD_SSH_PROXY_SUFFIX", "").strip()
        if suffix:
            user = f"{pod_id}-{suffix}"
    if not user:
        user = fetch_proxy_user_from_api(pod_id)
    if not user:
        # Optional fallback if runpodctl is on PATH.
        try:
            out = subprocess.check_output(
                ["runpodctl", "ssh", "info", pod_id],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            m = re.search(r"([\w-]+@ssh\.runpod\.io)", out)
            if m:
                user = m.group(1).split("@", 1)[0]
            else:
                m2 = re.search(r"ssh\s+([\w-]+)@ssh\.runpod\.io", out)
                if m2:
                    user = m2.group(1)
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            print(f"runpodctl ssh info failed: {exc}", file=sys.stderr)

    if not user:
        raise SystemExit(
            "Cannot resolve SSH proxy user from RunPod API. "
            "Check RUNPOD_API_KEY, or set vars.RUNPOD_SSH_PROXY_SUFFIX "
            f"(from Connect: {pod_id}-XXXXXXXX@ssh.runpod.io)."
        )

    if "@" in user:
        user = user.split("@", 1)[0]

    return [
        "ssh",
        "-tt",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "ConnectTimeout=30",
        "-i",
        key,
        f"{user}@ssh.runpod.io",
    ]


def tcp_works(ssh_argv: list[str]) -> bool:
    # Drop -tt for a quick non-interactive probe.
    probe = [a for a in ssh_argv if a != "-tt"] + ["true"]
    try:
        r = subprocess.run(probe, capture_output=True, timeout=20)
        return r.returncode == 0
    except subprocess.SubprocessError:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pod-id", required=True)
    p.add_argument("--pod-json", default="", help="raw JSON from RunPod API for TCP path")
    p.add_argument("--upload", default="", help="local:remote file path")
    p.add_argument("--script-file", required=True, help="local file with remote bash script")
    p.add_argument("--timeout", type=int, default=900)
    args = p.parse_args()

    if "RUNPOD_SSH_KEY_PATH" not in os.environ:
        raise SystemExit("RUNPOD_SSH_KEY_PATH is required")

    with open(args.script_file, encoding="utf-8") as f:
        remote_script = f.read()

    upload = None
    if args.upload:
        local, remote = args.upload.split(":", 1)
        upload = (local, remote)

    ssh_argv = None
    if args.pod_json:
        ssh_argv = resolve_tcp(args.pod_json)
        if ssh_argv and not tcp_works(ssh_argv):
            print("TCP SSH not reachable, falling back to proxy", file=sys.stderr)
            ssh_argv = None

    if ssh_argv is None:
        ssh_argv = resolve_proxy(args.pod_id)
        print(f"using proxy SSH: {ssh_argv[-1]}", file=sys.stderr)
    else:
        print(f"using TCP SSH: {ssh_argv[-1]} -p {ssh_argv[ssh_argv.index('-p')+1]}", file=sys.stderr)

    ssh_session(ssh_argv, remote_script, upload=upload, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
