#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright 2026, Intel Corporation


import re
import socket
import subprocess


def run(cmd, timeout=15):
    """Run a local shell command and return (stdout, stderr) as decoded strings."""
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return (
            proc.stdout.decode(errors='replace'),
            proc.stderr.decode(errors='replace'),
        )
    except subprocess.TimeoutExpired as e:
        return "", f"timeout after {timeout}s: {e}"
    except Exception as e:
        return "", f"exec error: {e}"


def section(title):
    print(f"\n--- {title} ---")


def check_gpu():
    section("GPU")

    # Try NVIDIA first
    out, _ = run("command -v nvidia-smi >/dev/null 2>&1 && "
                 "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if out.strip():
        gpus = [line.strip() for line in out.strip().splitlines() if line.strip()]
        print(f"[NVIDIA GPU count: {len(gpus)}]")
        for i, g in enumerate(gpus):
            print(f"  GPU {i}: {g}")
        return

    # Try Intel PVC / Data Center GPU via xpu-smi
    out, _ = run("command -v xpu-smi >/dev/null 2>&1 && xpu-smi discovery")
    if out.strip():
        print("[Intel xpu-smi discovery:]")
        print(out.strip())
        dev_count = len(re.findall(r"^\s*Device ID\s*:", out, flags=re.MULTILINE))
        if dev_count:
            print(f"[Intel GPU count: {dev_count}]")
        return

    # Fallback: use lspci to list any VGA / 3D / Display controllers
    out, _ = run("lspci -nn 2>/dev/null | grep -Ei 'vga|3d|display' || true")
    if out.strip():
        lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
        print(f"[GPU devices via lspci ({len(lines)} total):]")
        for l in lines:
            print(f"  {l}")

        joined = out.lower()
        detected = []
        if "pvc" in joined or "data center gpu max" in joined or "ponte vecchio" in joined:
            detected.append("Intel PVC (Data Center GPU Max)")
        if "h100" in joined:
            detected.append("NVIDIA H100")
        if "a100" in joined:
            detected.append("NVIDIA A100")
        if detected:
            print(f"[Detected: {', '.join(detected)}]")
        return

    print("[No GPU detected]")


def check_hostnamectl():
    section("hostnamectl")
    out, err = run("hostnamectl")
    if out.strip():
        print(out.strip())
    elif err.strip():
        print(f"[error] {err.strip()}")
    else:
        print("[no output]")


def check_storage():
    section("Storage")

    out, err = run("lsblk -d -b -o NAME,SIZE,MODEL,TYPE")
    if out.strip():
        lines = out.strip().splitlines()
        header, data = lines[0], lines[1:]
        disks = [l for l in data if l.strip().endswith("disk")]
        print(f"[disks ({len(disks)} total):]")
        print(header)
        total_bytes = 0
        for l in disks:
            print(l)
            parts = l.split()
            if len(parts) >= 2 and parts[1].isdigit():
                total_bytes += int(parts[1])
        if total_bytes:
            total_tb = total_bytes / (1024 ** 4)
            total_gb = total_bytes / (1024 ** 3)
            if total_tb >= 1:
                print(f"[total disk capacity: {total_tb:.2f} TiB ({total_gb:.0f} GiB)]")
            else:
                print(f"[total disk capacity: {total_gb:.2f} GiB]")
    elif err.strip():
        print(f"[lsblk error] {err.strip()}")

    out, _ = run("df -hT --total -x tmpfs -x devtmpfs -x squashfs -x overlay 2>/dev/null | tail -n +1")
    if out.strip():
        print("\n[df -hT (filesystems):]")
        print(out.strip())


def check_nic():
    section("NIC / NIC speed")

    ip_a_out, ip_a_err = run("ip a")
    if ip_a_out.strip():
        print("[ip a:]")
        print(ip_a_out.strip())
    elif ip_a_err.strip():
        print(f"[ip a error] {ip_a_err.strip()}")

    ip_br_out, _ = run("ip -4 -br addr show 2>/dev/null | awk '$1 != \"lo\"'")
    if ip_br_out.strip():
        print("\n[IPv4 addresses:]")
        print(ip_br_out.strip())

    print()

    out, err = run("ip -br link show 2>/dev/null | awk '$1 != \"lo\"'")
    if not out.strip():
        if err.strip():
            print(f"[ip link error] {err.strip()}")
        return

    ifaces = []
    print("[interfaces:]")
    print(out.strip())
    for line in out.strip().splitlines():
        parts = line.split()
        if parts:
            ifaces.append(parts[0].split('@')[0])

    print("\n[speed / model per interface:]")
    for iface in ifaces:
        speed_out, _ = run(
            f"( command -v ethtool >/dev/null 2>&1 && ethtool {iface} 2>/dev/null "
            f"| grep -E 'Speed:|Link detected:' ) || true")
        speed_line = ""
        if speed_out.strip():
            speed_line = " | ".join(l.strip() for l in speed_out.strip().splitlines())

        if not speed_line:
            sys_out, _ = run(f"cat /sys/class/net/{iface}/speed 2>/dev/null || true")
            if sys_out.strip():
                speed_line = f"Speed (sysfs): {sys_out.strip()} Mb/s"

        drv_out, _ = run(
            f"( command -v ethtool >/dev/null 2>&1 && ethtool -i {iface} 2>/dev/null "
            f"| grep -E '^(driver|bus-info):' ) || true")
        drv = ""
        bus_info = ""
        for line in drv_out.strip().splitlines():
            if line.startswith("driver:"):
                drv = line.split(":", 1)[1].strip()
            elif line.startswith("bus-info:"):
                bus_info = line.split(":", 1)[1].strip()

        if not bus_info:
            sys_bus, _ = run(
                f"basename $(readlink -f /sys/class/net/{iface}/device) 2>/dev/null || true")
            candidate = sys_bus.strip()
            if candidate and re.match(r"^[0-9a-fA-F]{4}:", candidate):
                bus_info = candidate

        model = ""
        if bus_info:
            lspci_out, _ = run(f"lspci -s {bus_info} 2>/dev/null || true")
            if lspci_out.strip():
                m = re.match(r"^\S+\s+[^:]+:\s*(.*)$", lspci_out.strip().splitlines()[0])
                model = m.group(1).strip() if m else lspci_out.strip().splitlines()[0]

        extras = []
        if drv:
            extras.append(f"driver: {drv}")
        if bus_info:
            extras.append(f"pci: {bus_info}")
        if model:
            extras.append(f"model: {model}")

        suffix = f"   [{' | '.join(extras)}]" if extras else ""
        print(f"  {iface}: {speed_line or 'unknown'}{suffix}")


def main():
    host = socket.gethostname()

    print(f"\n{'='*80}")
    print(f"Collecting info from local machine: {host}")
    print(f"{'='*80}")

    check_gpu()
    check_hostnamectl()
    check_storage()
    check_nic()

    print(f"\n{'='*80}")
    print(f"Done ({host})")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
