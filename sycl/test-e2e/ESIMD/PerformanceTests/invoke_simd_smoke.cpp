//==------------- invoke_simd_smoke.cpp  - DPC++ ESIMD on-device test----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu-intel-dg2 && level_zero
// UNSUPPORTED: windows

// RUN: mkdir -p %t.dir && %{build} -fsycl -fno-sycl-device-code-split-esimd -Xclang -fsycl-allow-func-ptr -o %t.dir/exec.out
// RUN: env  IGC_VCSaveStackCallLinkage=1 IGC_VCDirectCallsOnly=1 IGC_DumpToCustomDir=%t.dir IGC_ShaderDumpEnable=1 %{run} %t.dir/exec.out
// RUN: python3 %S/instruction_count.py %t.dir 149 _simd16_entry_0001.asm
// RUN: echo "Baseline from driver version 1.3.29735"

#include "../../InvokeSimd/invoke_simd_smoke.cpp"