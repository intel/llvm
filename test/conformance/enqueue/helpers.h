// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED
#define UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED

#include <cstring>
#include <uur/fixtures.h>

namespace uur {

struct test_parameters_t {
    std::string name;
    size_t src_size;
    size_t dst_size;
    ur_rect_offset_t src_origin;
    ur_rect_offset_t dst_origin;
    ur_rect_region_t region;
    size_t src_row_pitch;
    size_t src_slice_pitch;
    size_t dst_row_pitch;
    size_t dst_slice_pitch;
};

template <typename T>
inline std::string
printRectTestString(const testing::TestParamInfo<typename T::ParamType> &info) {
    // ParamType will be std::tuple<ur_device_handle_t, test_parameters_t>
    const auto device_handle = std::get<0>(info.param);
    const auto platform_device_name = GetPlatformAndDeviceName(device_handle);
    const auto &test_name = std::get<1>(info.param).name;
    return platform_device_name + "__" + test_name;
}

// Performs host side equivalent of urEnqueueMemBufferReadRect,
// urEnqueueMemBufferWriteRect and urEnqueueMemBufferCopyRect.
inline void copyRect(std::vector<uint8_t> src, ur_rect_offset_t src_offset,
                     ur_rect_offset_t dst_offset, ur_rect_region_t region,
                     size_t src_row_pitch, size_t src_slice_pitch,
                     size_t dst_row_pitch, size_t dst_slice_pitch,
                     std::vector<uint8_t> &dst) {
    const auto src_linear_offset = src_offset.x + src_offset.y * src_row_pitch +
                                   src_offset.z * src_slice_pitch;
    const auto src_start = src.data() + src_linear_offset;

    const auto dst_linear_offset = dst_offset.x + dst_offset.y * dst_row_pitch +
                                   dst_offset.z * dst_slice_pitch;
    const auto dst_start = dst.data() + dst_linear_offset;

    for (unsigned k = 0; k < region.depth; ++k) {
        const auto src_slice = src_start + k * src_slice_pitch;
        const auto dst_slice = dst_start + k * dst_slice_pitch;
        for (unsigned j = 0; j < region.height; ++j) {
            auto src_row = src_slice + j * src_row_pitch;
            auto dst_row = dst_slice + j * dst_row_pitch;
            std::memcpy(dst_row, src_row, region.width);
        }
    }
}

struct TestParameters2D {
    size_t pitch;
    size_t width;
    size_t height;
};

inline std::string USMKindToString(USMKind kind) {
    switch (kind) {
    case USMKind::Device:
        return "Device";
    case USMKind::Host:
        return "Host";
    case USMKind::Shared:
    default:
        return "Shared";
    }
}

template <typename T>
inline std::string
print2DTestString(const testing::TestParamInfo<typename T::ParamType> &info) {
    const auto device_handle = std::get<0>(info.param);
    const auto platform_device_name =
        uur::GetPlatformAndDeviceName(device_handle);
    std::stringstream test_name;
    auto src_kind = std::get<1>(std::get<1>(info.param));
    auto dst_kind = std::get<2>(std::get<1>(info.param));
    test_name << platform_device_name << "__pitch__"
              << std::get<0>(std::get<1>(info.param)).pitch << "__width__"
              << std::get<0>(std::get<1>(info.param)).width << "__height__"
              << std::get<0>(std::get<1>(info.param)).height << "__src__"
              << USMKindToString(src_kind) << "__dst__"
              << USMKindToString(dst_kind);
    return test_name.str();
}

} // namespace uur

#endif // UUR_ENQUEUE_RECT_HELPERS_H_INCLUDED
