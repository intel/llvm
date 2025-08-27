# System Performance Tuning Guide

This guide provides recommendations for optimizing system performance when running SYCL and Unified Runtime benchmarks.
For framework-specific information, see [README.md](README.md) and [CONTRIB.md](CONTRIB.md).

## Table of Contents

- [Overview](#overview)
- [System Configuration](#system-configuration)
- [CPU Tuning](#cpu-tuning)
- [GPU Configuration](#gpu-configuration)
- [Perf Configuration](#perf-configuration)
- [Environment Variables](#environment-variables)

## Overview

Performance benchmarking requires a stable and optimized system environment to produce reliable and reproducible results. This guide covers essential system tuning steps for reducing run-to-run variance in benchmark results.

## System Configuration

### Kernel Parameters

Add the following to `/etc/default/grub` in `GRUB_CMDLINE_LINUX`:
```
# Disable CPU frequency scaling
# intel_pstate=disable

# Isolate CPUs for benchmark workloads (example: reserve cores 2-7), preventing other processes
# from using them.
# isolcpus=2-7

GRUB_CMDLINE_LINUX="intel_pstate=disable isolcpus=2-7 <other_options>"
```

Update GRUB and reboot:
```bash
sudo update-grub
sudo reboot
```

## CPU Tuning

### CPU Frequency Scaling

The performance governor ensures that the CPU runs at maximum frequency.
```bash
# Set performance governor for all CPUs
sudo cpupower frequency-set --governor performance
# Apply changes to system
sudo sysctl --system

# Check current governor
sudo cpupower frequency-info
```

To preserve these settings after reboot, create a systemd service which runs the above commands at startup:
```bash
# Create a systemd service file
sudo vim /etc/systemd/system/cpupower_governor.service
```
Add the following content:
```
[Unit]
Description=Set CPU governor to Performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/cpupower frequency-set --governor performance && sysctl --system

[Install]
WantedBy=multi-user.target
```
Enable and start the service:
```bash
sudo systemctl enable cpupower_governor.service
sudo systemctl start cpupower_governor.service
```

### CPU Affinity

Bind benchmark processes to specific CPU cores to reduce context switching and improve cache locality.
Make sure that isolated CPUs are located on the same NUMA node as the GPU being used.
```bash
# Run benchmark on specific CPU cores
taskset -c 2-7 ./main.py ~/benchmarks_workdir/ --sycl ~/llvm/build/
```

## GPU Configuration

### GPU Frequency Control
Setting the GPU to run at maximum frequency can significantly improve benchmark performance and stability.

First, find which card relates to the GPU you want to tune (e.g., card1). List of known Device IDs for
Intel GPU cards can be found at https://dgpu-docs.intel.com/devices/hardware-table.html#gpus-with-supported-drivers.
```bash
# Print card1 Device ID
cat /sys/class/drm/card1/device/vendor  # Should be 0x8086 for Intel
cat /sys/class/drm/card1/device/device  # Device ID
```

Verify the max frequency is set to the true max. For Arc B580, the maximum frequency is 2850 MHz. To see this value, run “cat /sys/class/drm/card1/device/tile0/gt0/freq0/max_freq”. If the above value is not equal to the max frequency, set it as such:
```bash
# Arc B580 (Battlemage)
echo 2850 > /sys/class/drm/card1/device/tile0/gt0/freq0/max_freq

# Set the min frequency to the max frequency, so it is fixed
echo 2850 > /sys/class/drm/card1/device/tile0/gt0/freq0/min_freq
```

```bash
# Check GPU frequencies for GPU Max 1100 (Ponte Vecchio)
cat /sys/class/drm/card1/gt_max_freq_mhz
cat /sys/class/drm/card1/gt_min_freq_mhz

# Set maximum GPU frequency
max_freq=$(cat /sys/class/drm/card1/gt_max_freq_mhz)
echo $max_freq | sudo tee /sys/class/drm/card1/gt_min_freq_mhz
```

The result can be verified using tools such as oneprof or unitrace to track frequency over time for some arbitrary benchmark (many iterations of a small problem size is recommended). The frequency should remain fixed assuming thermal throttling does not occur.

## Driver version
Make sure you are using the latest driver (Ubuntu)
```bash
sudo apt update && sudo apt upgrade
```

## Perf Configuration

Some benchmarks (e.g., ur_submit_kernel) require `perf` to have proper permissions to sample hardware counters. This section covers the necessary setup steps.

### Installing Perf

First, ensure `perf` is installed for your kernel version. On Ubuntu/Debian:
```bash
sudo apt install linux-tools-$(uname -r) linux-tools-common
```

### Configuring Perf Permissions

By default, `perf` requires root privileges to access hardware performance counters.
Make the setting persistent across reboots by adding it to sysctl configuration:
```bash
echo 'kernel.perf_event_paranoid = -1' | sudo tee -a /etc/sysctl.d/99-perf.conf

# Apply immediately
sudo sysctl -p
```

### Verifying Perf Setup
Test that `perf` is working correctly:
```bash
perf record -e cycles sleep 1
perf report

# Test hardware counter access
perf stat -e cycles,instructions ls
```

**Note:** You may see warnings about kernel address maps being restricted or symbol resolution issues. These warnings don't prevent `perf` from collecting performance data - they only affect the quality of symbol resolution in kernel space. For user-space profiling (which covers most benchmark scenarios), `perf` will work correctly despite these warnings.

## Environment Variables

### Level Zero Environment Variables
Use GPU affinity to bind benchmarks to a specific GPU. Use CPUs from the same NUMA node as the GPU to reduce latency.
```bash
export ZE_AFFINITY_MASK=0
```

### SYCL Runtime Variables
For consistency, limit available devices to a specific gpu runtime. For Level Zero, it is recommended to use v2 version of the runtime library.
```bash
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export SYCL_UR_USE_LEVEL_ZERO_V2=1
```
