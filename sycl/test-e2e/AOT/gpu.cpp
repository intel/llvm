//==--- gpu.cpp - AOT compilation for gen devices using GEN compiler ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

// REQUIRES: ocloc, gpu, target-spir
// REQUIRES: intel-gpu-aot-targets || !new-offload-model
//
// RUN: %clangxx -fsycl %{gpu_aot_opts} %S/Inputs/aot.cpp -o %t.out
// RUN: %{run} %t.out
