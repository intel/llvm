// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "uur/fixtures.h"

namespace uur {
template <>
std::string deviceTestWithParamPrinter<BoolTestParam>(
    const ::testing::TestParamInfo<std::tuple<DeviceTuple, BoolTestParam>>
        &info) {
    auto device = std::get<0>(info.param).device;
    auto param = std::get<1>(info.param);

    std::stringstream ss;
    ss << param.name << (param.value ? "Enabled" : "Disabled");
    return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
}

template <>
std::string deviceTestWithParamPrinter<SamplerCreateParamT>(
    const ::testing::TestParamInfo<
        std::tuple<DeviceTuple, uur::SamplerCreateParamT>> &info) {
    auto device = std::get<0>(info.param).device;
    auto param = std::get<1>(info.param);

    const auto normalized = std::get<0>(param);
    const auto addr_mode = std::get<1>(param);
    const auto filter_mode = std::get<2>(param);

    std::stringstream ss;

    if (normalized) {
        ss << "NORMALIZED_";
    } else {
        ss << "UNNORMALIZED_";
    }
    ss << addr_mode << "_" << filter_mode;
    return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
}

template <>
std::string deviceTestWithParamPrinter<ur_image_format_t>(
    const ::testing::TestParamInfo<std::tuple<DeviceTuple, ur_image_format_t>>
        &info) {
    auto device = std::get<0>(info.param).device;
    auto param = std::get<1>(info.param);
    auto ChannelOrder = param.channelOrder;
    auto ChannelType = param.channelType;

    std::stringstream ss;
    ss << ChannelOrder << "__" << ChannelType;
    return uur::GetPlatformAndDeviceName(device) + "__" + ss.str();
}
} // namespace uur
