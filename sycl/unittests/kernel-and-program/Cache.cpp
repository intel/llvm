//==--------- Cache.cpp --- kernel and program cache unit test -------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/pi.h"
#include "detail/context_impl.hpp"
#include "detail/kernel_program_cache.hpp"
#include "detail/program_impl.hpp"
#include <CL/sycl.hpp>
#include <helpers/PiMock.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <iostream>

using namespace sycl;

class TestKernel {
public:
  void operator()(cl::sycl::item<1>){};
};

class TestKernel2 {
public:
  void operator()(cl::sycl::item<1>){};
};

struct TestCtx {
  detail::pi::PiContext context;
};

std::unique_ptr<TestCtx> globalCtx;

static pi_result redefinedProgramCreateWithSource(pi_context context,
                                                  pi_uint32 count,
                                                  const char **strings,
                                                  const size_t *lengths,
                                                  pi_program *ret_program) {
  return PI_SUCCESS;
}

static pi_result
redefinedProgramBuild(pi_program program, pi_uint32 num_devices,
                      const pi_device *device_list, const char *options,
                      void (*pfn_notify)(pi_program program, void *user_data),
                      void *user_data) {
  return PI_SUCCESS;
}

static pi_result redefinedProgramCompile(
    pi_program program, pi_uint32 num_devices, const pi_device *device_list,
    const char *options, pi_uint32 num_input_headers,
    const pi_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(pi_program program, void *user_data), void *user_data) {
  return PI_SUCCESS;
}

static pi_result
redefinedProgramLink(pi_context context, pi_uint32 num_devices,
                     const pi_device *device_list, const char *options,
                     pi_uint32 num_input_programs,
                     const pi_program *input_programs,
                     void (*pfn_notify)(pi_program program, void *user_data),
                     void *user_data, pi_program *ret_program) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelCreate(pi_program program,
                                       const char *kernel_name,
                                       pi_kernel *ret_kernel) {
  return PI_SUCCESS;
}

static pi_result redefinedKernelRetain(pi_kernel kernel) { return PI_SUCCESS; }

static pi_result redefinedKernelRelease(pi_kernel kernel) { return PI_SUCCESS; }

static pi_result redefinedKernelGetInfo(pi_kernel kernel,
                                        pi_kernel_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret) {
  if (param_name == PI_KERNEL_INFO_CONTEXT) {
    auto ctx = reinterpret_cast<detail::pi::PiContext *>(param_value);
    *ctx = globalCtx->context;
  }

  return PI_SUCCESS;
}

TEST(KernelAndProgramCache, ProgramSourceNegativeBuild) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  Mock.redefine<detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_source("");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for source programs";
}

TEST(KernelAndProgramCache, ProgramSourceNegativeBuildWithOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  Mock.redefine<detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_source("", "-g");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for source programs";
}

TEST(KernelAndProgramCache, ProgramSourceNegativeCompileAndLink) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_source("");
  Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for source programs";
}

TEST(KernelAndProgramCache, ProgramSourceNegativeCompileAndLinkWithOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_source("");
  Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for source programs";
}

TEST(KernelAndProgramCache, ProgramBuildPositive) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 1) << "Expect non-empty cache for programs";
}

TEST(KernelAndProgramCache, ProgramBuildNegativeBuildOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>("-g");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for programs";
}

TEST(KernelAndProgramCache, ProgramBuildNegativeCompileOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>("-g");
  Prg.link();
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for programs";
}

TEST(KernelAndProgramCache, ProgramBuildNegativeLinkOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>();
  Prg.link("-g");
  auto CtxImpl = detail::getSyclObjImpl(Ctx);
  detail::KernelProgramCache::ProgramCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireCachedPrograms().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for programs";
}

TEST(KernelAndProgramCache, KernelCachePositive) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>();
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 1) << "Expect non-empty cache for kernels";
}

TEST(KernelAndProgramCache, KernelCacheNegativeBuildOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_kernel_type<TestKernel>("-g");
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for kernels";
}
TEST(KernelAndProgramCache, KernelCacheNegativeCompileOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>("-g");
  Prg.link();
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for kernels";
}

TEST(KernelAndProgramCache, KernelCacheNegativeLinkOpts) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.compile_with_kernel_type<TestKernel>();
  Prg.link("-g");
  kernel Ker = Prg.get_kernel<TestKernel>();
  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for kernels";
}

TEST(KernelAndProgramCache, KernelCacheNegativeLinkedProgs) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piProgramCompile>(redefinedProgramCompile);
  Mock.redefine<detail::PiApiKind::piProgramLink>(redefinedProgramLink);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg1{Ctx};
  program Prg2{Ctx};

  Prg1.compile_with_kernel_type<TestKernel>();
  Prg2.compile_with_kernel_type<TestKernel2>();
  program Prg({Prg1, Prg2});
  kernel Ker = Prg.get_kernel<TestKernel>();

  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for kernels";
}

TEST(KernelAndProgramCache, KernelCacheNegativeSource) {
  platform Plt{default_selector()};
  if (Plt.is_host() || Plt.get_backend() != backend::opencl) {
    std::clog << "This test is only supported on OpenCL devices\n";
    return;
  }

  unittest::PiMock Mock(Plt);
  Mock.redefine<detail::PiApiKind::piclProgramCreateWithSource>(
      redefinedProgramCreateWithSource);
  Mock.redefine<detail::PiApiKind::piProgramBuild>(redefinedProgramBuild);
  Mock.redefine<detail::PiApiKind::piKernelCreate>(redefinedKernelCreate);
  Mock.redefine<detail::PiApiKind::piKernelRetain>(redefinedKernelRetain);
  Mock.redefine<detail::PiApiKind::piKernelRelease>(redefinedKernelRelease);
  Mock.redefine<detail::PiApiKind::piKernelGetInfo>(redefinedKernelGetInfo);

  context Ctx{Plt};
  auto CtxImpl = detail::getSyclObjImpl(Ctx);

  globalCtx.reset(new TestCtx{CtxImpl->getHandleRef()});

  program Prg{Ctx};

  Prg.build_with_source("");
  kernel Ker = Prg.get_kernel("test");

  detail::KernelProgramCache::KernelCacheT &Cache =
      CtxImpl->getKernelProgramCache().acquireKernelsPerProgramCache().get();
  EXPECT_EQ(Cache.size(), 0) << "Expect empty cache for kernels";
}
