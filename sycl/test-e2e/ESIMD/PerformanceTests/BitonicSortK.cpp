//==---------------- BitonicSortK.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 && level_zero
// UNSUPPORTED: windows

// RUN: mkdir -p %t.dir && %{build} -o %t.dir/exec.out
// RUN: env IGC_DumpToCustomDir=%t.dir IGC_ShaderDumpEnable=1 %{run} %t.dir/exec.out
// RUN: python3 %S/instruction_count.py %t.dir 2914 ZTSZZN11BitonicSort5SolveEPjS0_jENKUlRN4sycl3_V17handlerEE0_clES4_E5Merge.asm
// RUN: echo "Baseline from driver version 1.3.30872"

#include "../BitonicSortK.cpp"
