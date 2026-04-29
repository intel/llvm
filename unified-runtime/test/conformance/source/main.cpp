// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/environment.h>

int main(int argc, char **argv) {
#ifdef KERNELS_ENVIRONMENT
  auto *environment =
      new uur::KernelsEnvironment(argc, argv, KERNELS_DEFAULT_DIR);
#elif DEVICES_ENVIRONMENT
  auto *environment = new uur::DevicesEnvironment();
#elif PLATFORM_ENVIRONMENT
  auto *environment = new uur::PlatformEnvironment();
#else
  auto *environment = new uur::AdapterEnvironment();
#endif
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(environment);
  return RUN_ALL_TESTS();
}
