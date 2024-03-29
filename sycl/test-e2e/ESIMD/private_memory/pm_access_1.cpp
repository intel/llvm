//==--------------- pm_access_1.cpp - DPC++ ESIMD on-device test ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %clangxx -fsycl -Xs "-stateless-stack-mem-size=131072" -I%S/.. %S/Inputs/pm_common.cpp -o %t.out
// RUN: %{run} %t.out 1
