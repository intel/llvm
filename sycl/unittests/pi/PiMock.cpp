//==--------- PiMock.cpp --- A test for mock helper API's ------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <helpers/sycl_test.hpp>

#include <detail/queue_impl.hpp>

#include <gtest/gtest.h>

using namespace cl::sycl;

bool PiCalled = false;

static pi_result piProgramBuildRedefine(pi_program, pi_uint32,
                                        const pi_device *, const char *,
                                        void (*)(pi_program, void *), void *) {
  PiCalled = true;
  return PI_INVALID_BINARY;
}

pi_result piKernelCreateRedefine(pi_program, const char *, pi_kernel *) {
  PiCalled = true;
  return PI_INVALID_DEVICE;
}

SYCL_TEST(PiMockTest, RedefineAPI) {
  device Dev{default_selector()};
  if (Dev.is_host()) {
    std::cerr << "Not run due to host-only environment\n";
    return;
  }

  using namespace sycl::unittest;

  const auto &MockPiPlugin =
      detail::getSyclObjImpl(Dev.get_platform())->getPlugin().getPiPlugin();
  const auto &Table = MockPiPlugin.PiFunctionTable;

  // Pass a function pointer
  PiCalled = false;
  redefine<detail::PiApiKind::piProgramBuild>(piProgramBuildRedefine);
  ASSERT_FALSE(PiCalled);
  Table.piProgramBuild(nullptr, 0, nullptr, nullptr, nullptr, nullptr);
  EXPECT_TRUE(PiCalled)
      << "Function redefinition didn't propagate to the mock plugin";

  // Pass a std::function
  PiCalled = false;
  redefine<detail::PiApiKind::piKernelCreate>({piKernelCreateRedefine});
  ASSERT_FALSE(PiCalled);
  Table.piKernelCreate(nullptr, nullptr, nullptr);
  EXPECT_TRUE(PiCalled)
      << "Function redefinition didn't propagate to the mock plugin";

  // Pass a captureless lambda
  PiCalled = false;
  redefine<detail::PiApiKind::piProgramRetain>([](pi_program) -> pi_result {
    PiCalled = true;
    return PI_SUCCESS;
  });
  ASSERT_FALSE(PiCalled);
  Table.piProgramRetain(nullptr);
  EXPECT_TRUE(PiCalled)
      << "Function redefinition didn't propagate to the mock plugin";

  // Pass a lambda with captured reference
  bool PiCalledCapture = false;
  redefine<detail::PiApiKind::piEventRetain>(
      [&PiCalledCapture](pi_event) -> pi_result {
        PiCalledCapture = true;
        return PI_SUCCESS;
      });
  Table.piEventRetain(nullptr);
  EXPECT_TRUE(PiCalledCapture)
      << "Function redefinition didn't propagate to the mock plugin";
}
