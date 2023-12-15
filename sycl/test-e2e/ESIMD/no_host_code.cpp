//==---------------- no_host_code.cpp  - DPC++ ESIMD on-device test --------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// RUN: %{build} -fno-sycl-esimd-build-host-code -o %t.out
// RUN: %{run} %t.out

#include "BitonicSortK.hpp"
