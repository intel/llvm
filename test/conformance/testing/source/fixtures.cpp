// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"

namespace uur {
template <>
std::string deviceTestWithParamPrinter<BoolTestParam>(
    const ::testing::TestParamInfo<
        std::tuple<ur_device_handle_t, BoolTestParam>> &info) {
    auto device = std::get<0>(info.param);
    auto param = std::get<1>(info.param);

    std::stringstream ss;
    ss << param.name << (param.value ? "Enabled" : "Disabled");
    return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
}
} // namespace uur
