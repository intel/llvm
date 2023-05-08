//==--------------- pm_access_2.cpp - DPC++ ESIMD on-device test ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// Temporarily disabled due to flaky behavior
// REQUIRES: TEMPORARY_DISABLED
// RUN: %clangxx -fsycl -Xs "-stateless-stack-mem-size=131072" -I%S/.. %S/Inputs/pm_common.cpp -o %t.out
// RUN: %{run} %t.out 2
