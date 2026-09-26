// REQUIRES: linux, gpu && level_zero
// REQUIRES: (arch-intel_gpu_pvc || gpu-intel-dg2)

// XFAIL: arch-intel_gpu_pvc && new-offload-model
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/23265

// RUN: %{run-aux} %{build} %device_asan_aot_gpu_flags -O0 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not --crash %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_aot_gpu_flags -O1 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not --crash %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_aot_gpu_flags -O2 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not --crash %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_aot_gpu_flags -O3 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not --crash %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp
