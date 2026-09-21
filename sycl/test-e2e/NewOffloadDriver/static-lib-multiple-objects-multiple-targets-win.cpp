//==- static-lib-multiple-objects-multiple-targets-win.cpp ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Windows counterpart of static-lib-multiple-objects-multiple-targets.cpp: same
// scenario, built in MSVC mode (clang-cl, /MD) with cl for the final host link.
// The kernels and main are reused via the #include below.

// REQUIRES: ocloc, target-spir, windows
// REQUIRES: arch-intel_gpu_dg2_g10 || arch-intel_gpu_dg2_g11 || arch-intel_gpu_dg2_g12 || arch-intel_gpu_bmg_g21

// DEFINE: %{clcxx} = %clangxx --driver-mode=cl -fsycl --offload-new-driver /MD -Wno-error=unused-command-line-argument

// RUN: %{clcxx} -fsycl-targets=spir64 -c %S/Inputs/static-lib-registerlib-jit.cpp -o %t.jit.obj
// RUN: %{clcxx} -fsycl-link -fsycl-targets=spir64 %t.jit.obj -o %t.jit.pre.obj
// RUN: %{clcxx} -fsycl-targets=intel_gpu_dg2 -c %S/Inputs/static-lib-registerlib-dg2.cpp -o %t.dg2.obj
// RUN: %{clcxx} -fsycl-link -fsycl-targets=intel_gpu_dg2 %t.dg2.obj -o %t.dg2.pre.obj
// RUN: %{clcxx} -fsycl-targets=intel_gpu_bmg_g21 -c %S/Inputs/static-lib-registerlib-bmg.cpp -o %t.bmg.obj
// RUN: %{clcxx} -fsycl-link -fsycl-targets=intel_gpu_bmg_g21 %t.bmg.obj -o %t.bmg.pre.obj

// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t.jit.obj %t.jit.pre.obj %t.dg2.obj %t.dg2.pre.obj %t.bmg.obj %t.bmg.pre.obj

// RUN: %clangxx --driver-mode=cl /MD /std:c++17 %sycl_include -c %s -o %t.main.obj
// RUN: cl -nologo /MD %t.main.obj -Fe%t.exe -link %t.a /defaultlib:%sycl_static_libs_dir/sycl.lib
// RUN: %{run} %t.exe

// Reuse the kernels and main() from the Linux test.
#include "static-lib-multiple-objects-multiple-targets.cpp"
