// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
//==------------- kernel_cache.cpp - SYCL kernel/program test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

class Functor {
public:
  void operator()(cl::sycl::item<1> Item) { (void)Item; }
};

struct TestContext {
  int Data;
  cl::sycl::queue Queue;
  cl::sycl::buffer<int, 1> Buf;

  TestContext() : Data(0), Buf(&Data, cl::sycl::range<1>(1)) {}

  cl::sycl::program
  getProgram(const cl::sycl::string_class &BuildOptions = "") {
    cl::sycl::program Prog(Queue.get_context());

    Prog.build_with_kernel_type<class SingleTask>(BuildOptions);

    assert(Prog.get_state() == cl::sycl::program_state::linked &&
           "Linked state was expected");

    assert(Prog.has_kernel<class SingleTask>() &&
           "Expecting SingleTask kernel exists");

    return std::move(Prog);
  }

  cl::sycl::program getCompiledProgram() {
    cl::sycl::program Prog(Queue.get_context());

    Prog.compile_with_kernel_type<class SingleTask>();

    assert(Prog.get_state() == cl::sycl::program_state::compiled &&
           "Compiled state was expected");

    return std::move(Prog);
  }

  cl::sycl::kernel getKernel(cl::sycl::program &Prog) {
    auto Kernel = Prog.get_kernel<class SingleTask>();

    Queue.submit([&](cl::sycl::handler &CGH) {
      auto acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);
      CGH.single_task<class SingleTask>(Kernel, [=]() { acc[0] = acc[0] + 1; });
    });

    return std::move(Kernel);
  }
};

namespace pi = cl::sycl::detail::pi;
namespace RT = cl::sycl::RT;

static void testPositive() {
  TestContext TestCtx;

  auto Prog = TestCtx.getProgram();
  auto Kernel = TestCtx.getKernel(Prog);

  if (!TestCtx.Queue.is_host()) {
    auto *CLProg = cl::sycl::detail::getSyclObjImpl(Prog)->getHandleRef();
    auto *CLKernel = cl::sycl::detail::getSyclObjImpl(Kernel)->getHandleRef();

    auto *Ctx = cl::sycl::detail::getRawSyclObjImpl(Prog.get_context());

    assert(Ctx->getCachedPrograms().size() == 1 &&
           "Expecting only a single element in program cache");
    assert(Ctx->getCachedPrograms().begin()->second ==
               pi::cast<pi_program>(CLProg) &&
           "Invalid data in programs cache");
    assert(Ctx->getCachedKernels().size() == 1 &&
           "Expecting only a single element in kernels cache");
    assert(Ctx->getCachedKernels().begin()->first ==
               pi::cast<pi_program>(CLProg) &&
           "Invalid program key in kernels cache");
    assert(Ctx->getCachedKernels().begin()->second.size() == 1 &&
           "Expecting only a single kernel for the program");
    assert(Ctx->getCachedKernels().begin()->second.begin()->second ==
               pi::cast<pi_kernel>(CLKernel) &&
           "Invalid data in kernels cache");
  }
}

void testNegativeLinkedProgram() {
  TestContext TestCtx;

  auto Prog1 = TestCtx.getCompiledProgram();
  auto Prog2 = TestCtx.getCompiledProgram();

  auto LinkedProg = cl::sycl::program({Prog1, Prog2});

  auto Kernel = TestCtx.getKernel(LinkedProg);

  if (!TestCtx.Queue.is_host()) {
    auto *Ctx = cl::sycl::detail::getRawSyclObjImpl(LinkedProg.get_context());

    assert(Ctx->getCachedKernels().size() == 0 &&
           "Unexpected data in kernels cache");
  }
}

void testNegativeOCLProgram() {
  TestContext TestCtx;

  auto SyclProg = TestCtx.getProgram();

  auto OclProg = cl::sycl::program(TestCtx.Queue.get_context(), SyclProg.get());

  auto Kernel = TestCtx.getKernel(OclProg);

  if (!TestCtx.Queue.is_host()) {
    auto *Ctx = cl::sycl::detail::getRawSyclObjImpl(OclProg.get_context());

    assert(Ctx->getCachedKernels().size() == 0 &&
           "Unexpected data in kernels cache");
  }
}

void testNegativeCustomBuildOptions() {
  TestContext TestCtx;

  auto Prog = TestCtx.getProgram("-g");
  auto Kernel = TestCtx.getKernel(Prog);

  if (!TestCtx.Queue.is_host()) {
    auto *Ctx = cl::sycl::detail::getRawSyclObjImpl(Prog.get_context());
    assert(Ctx->getCachedPrograms().size() == 0 &&
           "Unexpected data in programs cache");
    assert(Ctx->getCachedKernels().size() == 0 &&
           "Unexpected data in kernels cache");
  }
}

int main() {
  testPositive();
  testNegativeLinkedProgram();
  testNegativeOCLProgram();
  testNegativeCustomBuildOptions();

  return 0;
}
