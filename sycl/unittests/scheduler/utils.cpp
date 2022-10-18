//==--------------- utils.cpp --- Scheduler unit tests ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTestUtils.hpp"

void addEdge(sycl::detail::Command *User, sycl::detail::Command *Dep,
             sycl::detail::AllocaCommandBase *Alloca) {
  std::vector<sycl::detail::Command *> ToCleanUp;
  (void)User->addDep(sycl::detail::DepDesc{Dep, User->getRequirement(), Alloca},
                     ToCleanUp);
  Dep->addUser(User);
}

sycl::detail::Requirement getMockRequirement() {
  return {/*Offset*/ {0, 0, 0},
          /*AccessRange*/ {0, 0, 0},
          /*MemoryRange*/ {0, 0, 0},
          /*AccessMode*/ sycl::access::mode::read_write,
          /*SYCLMemObj*/ nullptr,
          /*Dims*/ 0,
          /*ElementSize*/ 0};
}
