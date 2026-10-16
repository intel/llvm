//==- static-lib-multiple-objects-multiple-targets-lto.cpp--==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------===//
// Same case as static-lib-multiple-objects-multiple-targets.cpp but using LTO
// objects.

// REQUIRES: ocloc, target-spir, linux, lld
// REQUIRES: gpu-intel-dg2 || arch-intel_gpu_bmg_g21

// DEFINE: %{ond} = --offload-new-driver -Wno-error=unused-command-line-argument

// RUN: %clangxx -fsycl %{ond} -flto -fsycl-targets=spir64 -fPIC -c %S/Inputs/static-lib-registerlib-jit.cpp -o %t.jit.o
// RUN: %clangxx -fsycl %{ond} -flto -fsycl-link -fsycl-targets=spir64 -fPIC %t.jit.o -o %t.jit.pre.o
// RUN: %clangxx -fsycl %{ond} -flto -fsycl-targets=intel_gpu_dg2 -fPIC -c %S/Inputs/static-lib-registerlib-dg2.cpp -o %t.dg2.o
// RUN: %clangxx -fsycl %{ond} -flto -fsycl-link -fsycl-targets=intel_gpu_dg2 -fPIC %t.dg2.o -o %t.dg2.pre.o
// RUN: %clangxx -fsycl %{ond} -flto -fsycl-targets=intel_gpu_bmg_g21 -fPIC -c %S/Inputs/static-lib-registerlib-bmg.cpp -o %t.bmg.o
// RUN: %clangxx -fsycl %{ond} -flto -fsycl-link -fsycl-targets=intel_gpu_bmg_g21 -fPIC %t.bmg.o -o %t.bmg.pre.o

// RUN: rm -f %t.a
// RUN: llvm-ar crv %t.a %t.jit.o %t.jit.pre.o %t.dg2.o %t.dg2.pre.o %t.bmg.o %t.bmg.pre.o

// RUN: %clangxx %{ond} -flto -Wno-error=unused-command-line-argument %sycl_include %cxx_std_optionc++17 -fPIC -c %s -o %t.main.o

// Note this is a non-SYCL compile.
// RUN: %clangxx %{ond} -flto -fuse-ld=lld %t.main.o %t.a -L%sycl_libs_dir -lsycl -Wl,-rpath,%sycl_libs_dir -o %t.exe

// RUN: %{run} %t.exe

#include "static-lib-multiple-objects-multiple-targets.cpp"
