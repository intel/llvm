// REQUIRES: linux, ocloc, gpu && level_zero
// REQUIRES: build-and-run-mode

// RUN: %{build} %device_asan_gpu_aot_flag %S/kernel-filter.cpp -O2 -fsanitize-ignorelist=%p/ignorelist.txt -o %t1
// RUN: %{run} %t1 2>&1 | FileCheck %S/kernel-filter.cpp
