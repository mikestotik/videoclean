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


def detect_ssh_proxy_failure(text: str) -> Optional[str]:
    """Return a short reason when the RunPod SSH proxy refused the session."""
    low = (text or "").lower()
    if "container not found" in low:
        return "container not found"
    if "no such pod" in low or "pod not found" in low:
        return "pod not found"
    return None


def _wait_prompt(fd: int, buf: list[str], timeout: float = 45) -> None:
    end = time.time() + timeout
    while time.time() < end:
        _pump(fd, buf, 0.8)
        joined = "".join(buf)
        fatal = detect_ssh_proxy_failure(joined)
        if fatal:
            raise RuntimeError(
                f"RunPod SSH proxy: {fatal}. "
                "Pod may be restarting, or RUNPOD_SSH_PROXY_SUFFIX is stale — "
                "check Connect tab: ssh <podId>-XXXXXXXX@ssh.runpod.io"
            )
        tail = joined[-400:]
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

        # Always print the marker so a failed health check cannot hang Actions
        # waiting for DONE until --timeout.
        wrapped = (
            "set +e\n"
            + remote_script.rstrip()
            + f"\n_ec=$?\necho {marker}\nexit 0\n"
        )
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


def _extract_proxy_user(text: str) -> str:
    m = re.search(r"([\w-]+)@ssh\.runpod\.io", text)
    if m:
        return m.group(1)
    m = re.search(r'"username"\s*:\s*"([\w-]+)"', text)
    if m and "ssh" in text.lower():
        return m.group(1)
    return ""


def fetch_proxy_user_runpodctl(pod_id: str) -> str:
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    env = os.environ.copy()
    if api_key:
        env["RUNPOD_API_KEY"] = api_key
    try:
        out = subprocess.check_output(
            ["runpodctl", "ssh", "info", pod_id, "-o", "json"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=45,
            env=env,
        )
    except FileNotFoundError:
        print("runpodctl not on PATH", file=sys.stderr)
        return ""
    except subprocess.SubprocessError as exc:
        print(f"runpodctl ssh info failed: {exc}", file=sys.stderr)
        return ""
    user = _extract_proxy_user(out)
    if not user:
        try:
            import json

            data = json.loads(out)
            # Tolerant to shape changes.
            if isinstance(data, dict):
                user = (
                    (data.get("proxy") or {}).get("username")
                    or data.get("username")
                    or data.get("user")
                    or ""
                )
                if not user:
                    cmd = data.get("command") or data.get("ssh") or ""
                    user = _extract_proxy_user(str(cmd))
        except Exception:
            user = _extract_proxy_user(out)
    if user:
        print(f"resolved proxy user from runpodctl: {user}", file=sys.stderr)
    return (user or "").strip()


def fetch_proxy_user_from_api(pod_id: str) -> str:
    """Try REST v2, then GraphQL — some API keys only work on one surface."""
    import json
    import urllib.error
    import urllib.request

    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        return ""

    # REST v2
    req = urllib.request.Request(
        f"https://api.runpod.io/v2/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            pod = json.loads(resp.read().decode())
        user = ((pod.get("ssh") or {}).get("proxy") or {}).get("username") or ""
        if user:
            print(f"resolved proxy user from API v2: {user}", file=sys.stderr)
            return user.strip()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"v2 pod lookup failed: {exc}", file=sys.stderr)

    # GraphQL (legacy key style as query param — still widely used)
    query = (
        "query Pod($id: String!) { pod(input: { podId: $id }) { id "
        "desiredStatus machine { podHostId } } }"
    )
    body = json.dumps({"query": query, "variables": {"id": pod_id}}).encode()
    gql_url = f"https://api.runpod.io/graphql?api_key={api_key}"
    req = urllib.request.Request(
        gql_url,
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
        # GraphQL does not always expose proxy username; host id is sometimes used.
        pod = ((payload.get("data") or {}).get("pod")) or {}
        host = ((pod.get("machine") or {}).get("podHostId") or "").strip()
        # Historical Connect format: <podId>-<hostSuffix>@ssh.runpod.io
        if host:
            # host looks like "xxxx-6441174d" or similar; keep last 8 hex if present.
            m = re.search(r"([0-9a-f]{8})$", host, re.I)
            if m:
                user = f"{pod_id}-{m.group(1)}"
                print(f"resolved proxy user from GraphQL host id: {user}", file=sys.stderr)
                return user
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"graphql pod lookup failed: {exc}", file=sys.stderr)

    return ""


def proxy_user_candidates(pod_id: str) -> list[str]:
    """Ordered unique proxy usernames. Live sources first; suffix is a fallback."""
    ordered: list[str] = []

    def add(user: str, *, label: str) -> None:
        user = (user or "").strip()
        if not user:
            return
        if "@" in user:
            user = user.split("@", 1)[0]
        if user in ordered:
            return
        ordered.append(user)
        print(f"proxy candidate ({label}): {user}", file=sys.stderr)

    explicit = os.environ.get("RUNPOD_SSH_PROXY_USER", "").strip()
    if explicit:
        add(explicit, label="RUNPOD_SSH_PROXY_USER")
    add(fetch_proxy_user_runpodctl(pod_id), label="runpodctl")
    add(fetch_proxy_user_from_api(pod_id), label="api")
    suffix = os.environ.get("RUNPOD_SSH_PROXY_SUFFIX", "").strip()
    if suffix:
        add(f"{pod_id}-{suffix}", label="RUNPOD_SSH_PROXY_SUFFIX")
    return ordered


def _proxy_ssh_argv(user: str) -> list[str]:
    key = os.environ["RUNPOD_SSH_KEY_PATH"]
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


def resolve_proxy(pod_id: str) -> list[str]:
    users = proxy_user_candidates(pod_id)
    if not users:
        raise SystemExit(
            "Cannot resolve SSH proxy user. Install/run runpodctl, or set "
            "GitHub Actions variable RUNPOD_SSH_PROXY_SUFFIX to XXXXXXXX from "
            f"Connect tab command: ssh {pod_id}-XXXXXXXX@ssh.runpod.io"
        )
    return _proxy_ssh_argv(users[0])


def tcp_works(ssh_argv: list[str]) -> bool:
    # Drop -tt for a quick non-interactive probe.
    probe = [a for a in ssh_argv if a != "-tt"] + ["true"]
    try:
        r = subprocess.run(probe, capture_output=True, timeout=20)
        return r.returncode == 0
    except subprocess.SubprocessError:
        return False


def proxy_probe(ssh_argv: list[str]) -> tuple[bool, str]:
    """Non-interactive probe. Returns (ok, combined output)."""
    probe = [a for a in ssh_argv if a != "-tt"] + ["true"]
    try:
        r = subprocess.run(probe, capture_output=True, timeout=25)
    except subprocess.SubprocessError as exc:
        return False, str(exc)
    text = ((r.stdout or b"") + (r.stderr or b"")).decode(errors="replace")
    if r.returncode == 0:
        return True, text
    return False, text


def resolve_working_proxy(pod_id: str) -> list[str]:
    """Try live usernames, then the Actions suffix. Fail with a clear reason."""
    users = proxy_user_candidates(pod_id)
    if not users:
        raise SystemExit(
            "Cannot resolve SSH proxy user. Install/run runpodctl, or set "
            "GitHub Actions variable RUNPOD_SSH_PROXY_SUFFIX to XXXXXXXX from "
            f"Connect tab command: ssh {pod_id}-XXXXXXXX@ssh.runpod.io"
        )
    errors: list[str] = []
    for user in users:
        argv = _proxy_ssh_argv(user)
        print(f"probing proxy SSH: {user}@ssh.runpod.io", file=sys.stderr)
        ok, text = proxy_probe(argv)
        if ok:
            print(f"using proxy SSH: {user}@ssh.runpod.io", file=sys.stderr)
            return argv
        fatal = detect_ssh_proxy_failure(text) or f"exit non-zero ({text.strip()[:160]})"
        print(f"proxy probe failed for {user}: {fatal}", file=sys.stderr)
        errors.append(f"{user}: {fatal}")
    joined = "; ".join(errors)
    raise SystemExit(
        "No working RunPod SSH proxy user. "
        f"Tried: {joined}. "
        "Open the pod Connect tab and refresh GitHub variable RUNPOD_SSH_PROXY_SUFFIX "
        f"from ssh {pod_id}-XXXXXXXX@ssh.runpod.io (or wait until the container is up)."
    )


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
        ssh_argv = resolve_working_proxy(args.pod_id)
    else:
        print(f"using TCP SSH: {ssh_argv[-1]} -p {ssh_argv[ssh_argv.index('-p')+1]}", file=sys.stderr)

    ssh_session(ssh_argv, remote_script, upload=upload, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
