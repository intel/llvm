// REQUIRES: linux, opencl-aot, cpu

// RUN: %{run-aux} %{build} %device_asan_aot_flags %S/kernel-filter.cpp -g -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} %t1 2>&1 | FileCheck %S/kernel-filter.cpp --check-prefixes CHECK-IGNORE
// RUN: %{run-aux} %{build} %device_asan_aot_flags %S/kernel-filter.cpp -g -O2 -o %t2
// RUN: %{run} not %t2 2>&1 | FileCheck %S/kernel-filter.cpp
