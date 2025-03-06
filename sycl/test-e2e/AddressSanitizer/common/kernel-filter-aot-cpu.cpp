// REQUIRES: linux, opencl-aot, cpu

// RUN: %{run-aux} %{build} %device_asan_aot_flags %S/kernel-filter.cpp -g -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} not %t1 2>&1 | FileCheck %S/kernel-filter.cpp
