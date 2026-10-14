"""Per-user Docker image overlay with host uid/gid."""

from __future__ import annotations

import grp
import os
import subprocess
import sys
from getpass import getuser


def generate_user_dockerfile(base_image: str) -> bytes:
    try:
        host_input_group_id = grp.getgrnam("input").gr_gid
    except KeyError:
        host_input_group_id = 65534

    user = getuser()
    uid = os.geteuid()
    gid = os.getgid()

    content = f"""
FROM {base_image}
RUN userdel -r ubuntu 2>/dev/null || true; \\
    groupadd --gid {gid} --non-unique {user} || true; \\
    useradd --create-home --shell /bin/bash --uid {uid} --gid {gid} --non-unique {user}; \\
    usermod -aG sudo {user} || true; \\
    groupadd --non-unique -g {host_input_group_id} host_machine_input || true; \\
    usermod -aG host_machine_input {user} || true; \\
    echo "{user} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
"""
    return content.encode("utf-8")


def build_user_image(base_image: str, user_image: str) -> int:
    print(f"Building user image {user_image} from {base_image}...")
    result = subprocess.run(
        ["docker", "build", "-t", user_image, "-"],
        input=generate_user_dockerfile(base_image),
        check=False,
    )
    if result.returncode != 0:
        print(f"User image build failed for {user_image}", file=sys.stderr)
        return result.returncode
    print(f"Built {user_image}")
    return 0
