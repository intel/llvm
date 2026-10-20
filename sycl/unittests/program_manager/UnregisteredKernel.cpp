//==------------------- UnregisteredKernel.cpp ---------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Launching a kernel whose device image was never registered (e.g. its TU was
// compiled with -fsycl-host-only) must throw a sycl::exception rather than
// assert or crash inside the ProgramManager lookup.
//
//===----------------------------------------------------------------------===//

#include <detail/program_manager/program_manager.hpp>
#include <helpers/MockKernelInfo.hpp>
#include <helpers/UrMock.hpp>
#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <string_view>

// Has a KernelInfo specialization but deliberately no MockDeviceImage.
class UnregisteredKernel;
MOCK_INTEGRATION_HEADER(UnregisteredKernel)

namespace {

void expectNoKernelNamed(const std::function<void()> &Launch,
                         const std::string &Name) {
  try {
    Launch();
    FAIL() << "expected sycl::exception for kernel " << Name;
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), sycl::make_error_code(sycl::errc::runtime));
    EXPECT_NE(std::string(E.what()).find("No kernel named " + Name),
              std::string::npos)
        << E.what();
  }
}

} // namespace

TEST(UnregisteredKernel, SubmitThrows) {
  sycl::unittest::UrMock<> Mock;
  sycl::queue Q{sycl::platform().get_devices()[0]};

  auto Submit = [&] {
    Q.submit(
        [](sycl::handler &CGH) { CGH.single_task<UnregisteredKernel>([] {}); });
  };
  expectNoKernelNamed(Submit, "UnregisteredKernel");
  // The header caches the lookup in a function-local static; a failed
  // initialization must not poison later attempts.
  expectNoKernelNamed(Submit, "UnregisteredKernel");
}

TEST(UnregisteredKernel, LookupByNameThrows) {
  sycl::unittest::UrMock<> Mock;
  auto &PM = sycl::detail::ProgramManager::getInstance();
  const std::string Name = "NoSuchFreeFunctionKernel";

  expectNoKernelNamed([&] { PM.getDeviceKernelInfo(std::string_view(Name)); },
                      Name);
  EXPECT_EQ(PM.tryGetDeviceKernelInfo(Name), nullptr);
}
