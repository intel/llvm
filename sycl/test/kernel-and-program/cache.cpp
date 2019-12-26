// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
//==---------------- cache.cpp - SYCL kernel/program test ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

namespace RT = cl::sycl::RT;
namespace detail = cl::sycl::detail;
namespace pi = detail::pi;

using ProgramCacheT = detail::KernelProgramCache::ProgramCacheT;
using KernelCacheT = detail::KernelProgramCache::KernelCacheT;

#define KERNEL_NAME_SRC "kernel_source"
#define TEST_SOURCE                                                            \
  "kernel void " KERNEL_NAME_SRC "(global int* a) "                            \
  "{ a[get_global_id(0)] += 1; }\n"

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

  cl::sycl::program
  getProgramWSource(const cl::sycl::string_class &BuildOptions = "") {
    cl::sycl::program Prog(Queue.get_context());

    Prog.build_with_source(TEST_SOURCE, BuildOptions);

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

  cl::sycl::program
  getCompiledAndLinkedProgram(const cl::sycl::string_class &CompileOptions = "",
                              const cl::sycl::string_class &LinkOptions = "") {
    cl::sycl::program Prog(Queue.get_context());

    Prog.compile_with_kernel_type<class SingleTask>(CompileOptions);

    assert(Prog.get_state() == cl::sycl::program_state::compiled &&
           "Compiled state was expected");

    Prog.link(LinkOptions);

    assert(Prog.get_state() == cl::sycl::program_state::linked &&
           "Linked state was expected");

    return std::move(Prog);
  }

  cl::sycl::program getCompiledAndLinkedProgramWSource(
      const cl::sycl::string_class &CompileOptions = "",
      const cl::sycl::string_class &LinkOptions = "") {
    cl::sycl::program Prog(Queue.get_context());

    Prog.compile_with_source(TEST_SOURCE, CompileOptions);

    assert(Prog.get_state() == cl::sycl::program_state::compiled &&
           "Compiled state was expected");

    Prog.link(LinkOptions);

    assert(Prog.get_state() == cl::sycl::program_state::linked &&
           "Linked state was expected");

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

  cl::sycl::kernel getKernelWSource(cl::sycl::program &Prog) {
    auto Kernel = Prog.get_kernel(KERNEL_NAME_SRC);

    return std::move(Kernel);
  }
};

static void testProgramCachePositive() {
  TestContext TestCtx;

  auto Prog = TestCtx.getProgram();

  auto *CLProg = detail::getSyclObjImpl(Prog)->getHandleRef();

  auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
  detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
  const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

  assert(CachedPrograms.size() == 1 &&
         "Expecting only a single element in program cache");
  assert(CachedPrograms.begin()->second.Ptr.load() ==
             pi::cast<pi_program>(CLProg) &&
         "Invalid data in programs cache");
}

static void testProgramCacheNegativeCustomBuildOptions() {
  TestContext TestCtx;

  auto Prog = TestCtx.getProgram("-g");

  auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
  detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
  const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

  assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
}

static void testProgramCacheNegativeCompileLinkCustomOpts() {
  TestContext TestCtx;

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgram();

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }

  {
    auto Prog =
        TestCtx.getCompiledAndLinkedProgram("-g", "-cl-no-signed-zeroes");

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgram("", "-cl-no-signed-zeroes");

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgram("-g", "");

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }
}

static void testProgramCacheNegativeCompileLinkSource() {
  TestContext TestCtx;

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgramWSource();

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgramWSource(
        "-g", "-cl-no-signed-zeroes");

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }

  {
    auto Prog =
        TestCtx.getCompiledAndLinkedProgramWSource("", "-cl-no-signed-zeroes");

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgramWSource("-g", "");

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();

    assert(CachedPrograms.size() == 0 && "Expecting empty program cache");
  }
}

static void testKernelCachePositive() {
  TestContext TestCtx;

  auto Prog = TestCtx.getProgram();
  auto Kernel = TestCtx.getKernel(Prog);

  if (!TestCtx.Queue.is_host()) {
    auto *CLProg = detail::getSyclObjImpl(Prog)->getHandleRef();
    auto *CLKernel = detail::getSyclObjImpl(Kernel)->getHandleRef();

    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
    const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

    assert(CachedKernels.size() == 1 &&
           "Expecting only a single element in kernels cache");

    const auto &KernelsByNameIt = CachedKernels.begin();

    assert(KernelsByNameIt->first == pi::cast<pi_program>(CLProg) &&
           "Invalid program key in kernels cache");

    const auto &KernelsByName = KernelsByNameIt->second;

    assert(KernelsByName.size() == 1 &&
           "Expecting only a single kernel for the program");

    const auto &KernelWithBuildState = KernelsByName.begin()->second;

    assert(KernelWithBuildState.Ptr.load() == pi::cast<pi_kernel>(CLKernel) &&
           "Invalid data in kernels cache");
  }
}

void testKernelCacheNegativeLinkedProgram() {
  TestContext TestCtx;

  auto Prog1 = TestCtx.getCompiledProgram();
  auto Prog2 = TestCtx.getCompiledProgram();

  auto LinkedProg = cl::sycl::program({Prog1, Prog2});

  auto Kernel = TestCtx.getKernel(LinkedProg);

  if (!TestCtx.Queue.is_host()) {
    auto *Ctx = detail::getRawSyclObjImpl(LinkedProg.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
    const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

    assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
  }
}

void testKernelCacheNegativeOCLProgram() {
  TestContext TestCtx;

  auto SyclProg = TestCtx.getProgram();

  auto OclProg = cl::sycl::program(TestCtx.Queue.get_context(), SyclProg.get());

  auto Kernel = TestCtx.getKernel(OclProg);

  if (!TestCtx.Queue.is_host()) {
    auto *Ctx = detail::getRawSyclObjImpl(OclProg.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
    const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

    assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
  }
}

void testKernelCacheNegativeCustomBuildOptions() {
  TestContext TestCtx;

  auto Prog = TestCtx.getProgram("-g");
  auto Kernel = TestCtx.getKernel(Prog);

  if (!TestCtx.Queue.is_host()) {
    auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
    detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
    const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
    const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

    assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
  }
}

void testKernelCacheNegativeCompileLink() {
  TestContext TestCtx;

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgram();
    auto Kernel = TestCtx.getKernel(Prog);

    if (!TestCtx.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }

  {
    TestContext TestCtx1;
    auto Prog =
        TestCtx1.getCompiledAndLinkedProgram("-g", "-cl-no-signed-zeroes");
    auto Kernel = TestCtx1.getKernel(Prog);

    if (!TestCtx1.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgram("-g", "");
    auto Kernel = TestCtx.getKernel(Prog);

    if (!TestCtx.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgram("", "-cl-no-signed-zeroes");
    auto Kernel = TestCtx.getKernel(Prog);

    if (!TestCtx.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }
}

void testKernelCacheNegativeCompileLinkSource() {
  TestContext TestCtx;

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgramWSource();
    auto Kernel = TestCtx.getKernelWSource(Prog);

    if (!TestCtx.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgramWSource(
        "-g", "-cl-no-signed-zeroes");
    auto Kernel = TestCtx.getKernelWSource(Prog);

    if (!TestCtx.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }

  {
    auto Prog = TestCtx.getCompiledAndLinkedProgramWSource("-g", "");
    auto Kernel = TestCtx.getKernelWSource(Prog);

    if (!TestCtx.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }

  {
    auto Prog =
        TestCtx.getCompiledAndLinkedProgramWSource("", "-cl-no-signed-zeroes");
    auto Kernel = TestCtx.getKernelWSource(Prog);

    if (!TestCtx.Queue.is_host()) {
      auto *Ctx = detail::getRawSyclObjImpl(Prog.get_context());
      detail::KernelProgramCache &Cache = Ctx->getKernelProgramCache();
      const ProgramCacheT &CachedPrograms = Cache.acquireCachedPrograms().get();
      const KernelCacheT &CachedKernels = Cache.acquireKernelsPerProgramCache().get();

      assert(CachedKernels.size() == 0 && "Unexpected data in kernels cache");
    }
  }
}

int main() {
  testProgramCachePositive();
  testProgramCacheNegativeCustomBuildOptions();
  testProgramCacheNegativeCompileLinkCustomOpts();
  testProgramCacheNegativeCompileLinkSource();

  testKernelCachePositive();
  testKernelCacheNegativeLinkedProgram();
  testKernelCacheNegativeOCLProgram();
  testKernelCacheNegativeCustomBuildOptions();
  testKernelCacheNegativeCompileLink();
  testKernelCacheNegativeCompileLinkSource();

  return 0;
}
