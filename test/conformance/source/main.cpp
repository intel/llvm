// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <uur/environment.h>

int main(int argc, char **argv) {
#ifdef KERNELS_ENVIRONMENT
    auto *environment =
        new uur::KernelsEnvironment(argc, argv, KERNELS_DEFAULT_DIR);
#elif DEVICES_ENVIRONMENT
    auto *environment = new uur::DevicesEnvironment(argc, argv);
#elif PLATFORM_ENVIRONMENT
    auto *environment = new uur::PlatformEnvironment(argc, argv);
#else
    auto *environment = new uur::AdapterEnvironment();
#endif
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(environment);
    return RUN_ALL_TESTS();
}
