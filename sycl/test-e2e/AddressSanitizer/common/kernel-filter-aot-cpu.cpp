// REQUIRES: linux, opencl-aot, cpu
// REQUIRES: build-and-run-mode

// RUN: %{build} %device_asan_cpu_aot_flag %S/kernel-filter.cpp -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} %t1 2>&1 | FileCheck %S/kernel-filter.cpp
