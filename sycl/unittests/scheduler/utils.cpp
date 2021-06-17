//==--------------- utils.cpp --- Scheduler unit tests ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SchedulerTestUtils.hpp"

void addEdge(__sycl_internal::__v1::detail::Command *User, __sycl_internal::__v1::detail::Command *Dep,
             __sycl_internal::__v1::detail::AllocaCommandBase *Alloca) {
  (void)User->addDep(
      __sycl_internal::__v1::detail::DepDesc{Dep, User->getRequirement(), Alloca});
  Dep->addUser(User);
}

__sycl_internal::__v1::detail::Requirement getMockRequirement() {
  return {/*Offset*/ {0, 0, 0},
          /*AccessRange*/ {0, 0, 0},
          /*MemoryRange*/ {0, 0, 0},
          /*AccessMode*/ sycl::access::mode::read_write,
          /*SYCLMemObj*/ nullptr,
          /*Dims*/ 0,
          /*ElementSize*/ 0};
}
