// REQUIRES: linux, gpu && level_zero
// REQUIRES: (arch-intel_gpu_pvc || gpu-intel-dg2)

// RUN: %{run-aux} %{build} %device_asan_aot_flags %S/kernel-filter.cpp -g -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} not %t1 2>&1 | FileCheck %S/kernel-filter.cpp
