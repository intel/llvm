// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==--- program_manager.cpp - SYCL program manager test --------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <CL/sycl/detail/os_util.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <CL/sycl/detail/program_manager/program_manager.hpp>

#include <cassert>

using namespace cl::sycl;

int main() {
  context ContextFirst;
  context ContextSecond;

  auto &PM = detail::ProgramManager::getInstance();
  auto M = detail::OSUtil::ExeModuleHandle;

  const detail::RT::PiProgram ClProgramFirst = PM.getBuiltOpenCLProgram(M, ContextFirst);
  const detail::RT::PiProgram ClProgramSecond = PM.getBuiltOpenCLProgram(M, ContextSecond);
  // The check what getBuiltOpenCLProgram returns unique cl_program for unique
  // context
  assert(ClProgramFirst != ClProgramSecond);
  for (size_t i = 0; i < 10; ++i) {
    const detail::RT::PiProgram ClProgramFirstNew =
        PM.getBuiltOpenCLProgram(M, ContextFirst);
    const detail::RT::PiProgram ClProgramSecondNew =
        PM.getBuiltOpenCLProgram(M, ContextSecond);
    // The check what getBuiltOpenCLProgram returns the same program for the
    // same context each time
    assert(ClProgramFirst == ClProgramFirstNew);
    assert(ClProgramSecond == ClProgramSecondNew);
  }

  queue q;
  q.submit([&](handler &cgh) { cgh.single_task<class foo>([]() {}); });

  return 0;
}
