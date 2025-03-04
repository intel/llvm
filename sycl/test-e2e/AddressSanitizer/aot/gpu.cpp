// REQUIRES: linux, ocloc, gpu && level_zero
// REQUIRES: build-and-run-mode

// RUN: %{build} %device_asan_gpu_aot_flag -O0 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{build} %device_asan_gpu_aot_flag -O1 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{build} %device_asan_gpu_aot_flag -O2 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{build} %device_asan_gpu_aot_flag -O3 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp
