//==--------------- utils.cpp --- Scheduler unit tests ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTestUtils.hpp"

void addEdge(cl::sycl::detail::Command *User, cl::sycl::detail::Command *Dep,
             cl::sycl::detail::AllocaCommandBase *Alloca) {
  (void)User->addDep(
      cl::sycl::detail::DepDesc{Dep, User->getRequirement(), Alloca});
  Dep->addUser(User);
}

cl::sycl::detail::Requirement getMockRequirement() {
  return {/*Offset*/ {0, 0, 0},
          /*AccessRange*/ {0, 0, 0},
          /*MemoryRange*/ {0, 0, 0},
          /*AccessMode*/ cl::sycl::access::mode::read_write,
          /*SYCLMemObj*/ nullptr,
          /*Dims*/ 0,
          /*ElementSize*/ 0};
}
