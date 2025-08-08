// REQUIRES: linux, gpu && level_zero
// REQUIRES: arch-intel_gpu_pvc

// RUN: %{run-aux} %{build} %device_msan_aot_flags -O0 -g %S/check_divide.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/check_divide.cpp

// RUN: %{run-aux} %{build} %device_msan_aot_flags -O1 -g %S/check_divide.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/check_divide.cpp

// RUN: %{run-aux} %{build} %device_msan_aot_flags -O2 -g %S/check_divide.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/check_divide.cpp
