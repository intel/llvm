// REQUIRES: linux, ocloc, gpu && level_zero
// REQUIRES: build-and-run-mode
// REQUIRES: (arch-intel_gpu_pvc || gpu-intel-dg2)

// RUN: %{build} %device_asan_gpu_aot_flag %S/kernel-filter.cpp -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} %t1 2>&1 | FileCheck %S/kernel-filter.cpp
