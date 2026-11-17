// REQUIRES: linux, gpu && level_zero
// REQUIRES: arch-intel_gpu_bmg_g21

// BMG is the only AOT target here, so the driver links the AOT device sanitizer
// library rather than the JIT one it falls back to for multiple targets.

// RUN: %{run-aux} %{build} %device_asan_flags -fsycl-targets=intel_gpu_bmg_g21 -O0 -g %S/Inputs/host-usm-oob.cpp -o %t1.out
// RUN: %{run} not --crash %t1.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_flags -fsycl-targets=intel_gpu_bmg_g21 -O1 -g %S/Inputs/host-usm-oob.cpp -o %t2.out
// RUN: %{run} not --crash %t2.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_flags -fsycl-targets=intel_gpu_bmg_g21 -O2 -g %S/Inputs/host-usm-oob.cpp -o %t3.out
// RUN: %{run} not --crash %t3.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_flags -fsycl-targets=intel_gpu_bmg_g21 -O3 -g %S/Inputs/host-usm-oob.cpp -o %t4.out
// RUN: %{run} not --crash %t4.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp
