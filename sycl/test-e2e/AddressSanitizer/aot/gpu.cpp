// REQUIRES: linux, ocloc, gpu && level_zero
// REQUIRES: build-and-run-mode

// RUN: %clangxx  %device_asan_flags -O0 -g  -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %clangxx  %device_asan_flags -O1 -g  -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %clangxx  %device_asan_flags -O2 -g  -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp

// RUN: %clangxx  %device_asan_flags -O3 -g  -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen %gpu_aot_target_opts %S/Inputs/host-usm-oob.cpp -o %t.out
// RUN: %{run} not %t.out 2>&1 | FileCheck %S/Inputs/host-usm-oob.cpp
