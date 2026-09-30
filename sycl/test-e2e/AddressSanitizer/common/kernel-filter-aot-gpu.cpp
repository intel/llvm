// REQUIRES: linux, gpu && level_zero
// REQUIRES: (arch-intel_gpu_pvc || gpu-intel-dg2)

// XFAIL: arch-intel_gpu_pvc && new-offload-model
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23265

// RUN: %{run-aux} %{build} %device_asan_aot_gpu_flags %S/kernel-filter.cpp -g -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} %t1 2>&1 | FileCheck %S/kernel-filter.cpp --check-prefixes CHECK-IGNORE
// RUN: %{run-aux} %{build} %device_asan_aot_gpu_flags %S/kernel-filter.cpp -g -O2 -o %t2
// RUN: %{run} not --crash %t2 2>&1 | FileCheck %S/kernel-filter.cpp
