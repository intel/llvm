// Copyright (C) 2022-2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED
#define UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED

#include <uur/fixtures.h>
#include <uur/utils.h>

namespace uur {
namespace platform {

template <class T>
struct urPlatformTestWithParam
    : ::testing::Test,
      ::testing::WithParamInterface<std::tuple<ur_platform_handle_t, T>> {
    void SetUp() override { platform = std::get<0>(this->GetParam()); }
    const T &getParam() const { return std::get<1>(this->GetParam()); }
    ur_platform_handle_t platform;
};

template <class T>
std::string platformTestWithParamPrinter(
    const ::testing::TestParamInfo<std::tuple<ur_platform_handle_t, T>> &info) {
    auto platform = std::get<0>(info.param);
    auto param = std::get<1>(info.param);

    std::stringstream ss;
    ss << param;
    return uur::GetPlatformNameWithID(platform) + "__" +
           GTestSanitizeString(ss.str());
}

} // namespace platform
} // namespace uur

#define UUR_PLATFORM_TEST_SUITE_P(FIXTURE, VALUES, TYPE)                       \
    INSTANTIATE_TEST_SUITE_P(                                                  \
        , FIXTURE,                                                             \
        testing::Combine(                                                      \
            ::testing::ValuesIn(                                               \
                uur::PlatformEnvironment::instance->all_platforms),            \
            VALUES),                                                           \
        uur::platform::platformTestWithParamPrinter<TYPE>)

#endif // UR_CONFORMANCE_PLATFORM_FIXTURES_H_INCLUDED
