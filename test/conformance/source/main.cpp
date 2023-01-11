// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: MIT

#include <uur/environment.h>

int main(int argc, char **argv) {
#ifdef DEVICES_ENVIRONMENT
  auto *environment = new uur::DevicesEnvironment(argc, argv);
#endif
#ifdef PLATFORM_ENVIRONMENT
  auto *environment = new uur::PlatformEnvironment(argc, argv);
#endif
  ::testing::InitGoogleTest(&argc, argv);
#if defined(DEVICES_ENVIRONMENT) || defined(PLATFORM_ENVIRONMENT)
  ::testing::AddGlobalTestEnvironment(environment);
#endif
  return RUN_ALL_TESTS();
}
