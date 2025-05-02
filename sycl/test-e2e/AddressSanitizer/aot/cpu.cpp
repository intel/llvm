// REQUIRES: linux, opencl-aot, cpu

// RUN: %{run-aux} %{build} %device_asan_aot_flags -O0 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_aot_flags -O1 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_aot_flags -O2 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %{run-aux} %{build} %device_asan_aot_flags -O3 -g %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp
