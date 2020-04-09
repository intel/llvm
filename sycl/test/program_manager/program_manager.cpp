// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -I %sycl_source_dir %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// TODO rewrite as unit test
// XFAIL: *

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
#include <detail/program_manager/program_manager.hpp>

#include <cassert>
#include <map>

using namespace cl::sycl;

class KernelNameT;

int main() {
  context ContextFirst;
  context ContextSecond;

  auto &PM = detail::ProgramManager::getInstance();
  auto M = detail::OSUtil::ExeModuleHandle;

  string_class KernelNameStr = detail::KernelInfo<KernelNameT>::getName();
  const detail::RT::PiProgram ClProgramFirst =
      PM.getBuiltPIProgram(M, ContextFirst, KernelNameStr);
  const detail::RT::PiProgram ClProgramSecond =
      PM.getBuiltPIProgram(M, ContextSecond, KernelNameStr);
  // The check what getBuiltOpenCLProgram returns unique cl_program for unique
  // context
  assert(ClProgramFirst != ClProgramSecond);
  for (size_t i = 0; i < 10; ++i) {
    const detail::RT::PiProgram ClProgramFirstNew =
        PM.getBuiltPIProgram(M, ContextFirst, KernelNameStr);
    const detail::RT::PiProgram ClProgramSecondNew =
        PM.getBuiltPIProgram(M, ContextSecond, KernelNameStr);
    // The check what getBuiltOpenCLProgram returns the same program for the
    // same context each time
    assert(ClProgramFirst == ClProgramFirstNew);
    assert(ClProgramSecond == ClProgramSecondNew);
  }

  queue q;
  q.submit([&](handler &cgh) { cgh.single_task<KernelNameT>([]() {}); });

  return 0;
}
