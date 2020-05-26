//==------- device-code-split-utils.hpp --- Device code split unit tests ----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>
#include <gtest/gtest.h>
#include <helpers/PiMock.hpp>
#include <CL/sycl/detail/pi.hpp>
#include <map>

namespace DeviceCodeSplitTests {

std::map<std::string, bool> expectedKernel;

pi_result piextDeviceSelectBinaryRedefine(pi_device device,
                                           pi_device_binary *images,
                                           pi_uint32 num_images,
                                           pi_uint32 *selected_image_ind) {

    for (pi_uint32 i = 0; i < num_images; ++i) {
        auto RawImg = images[i];
        const _pi_offload_entry EntriesB = RawImg->EntriesBegin;
        const _pi_offload_entry EntriesE = RawImg->EntriesEnd;

        std::map<std::string, bool> hasKernel {{"File1Kern1", false },
                                               { "File1Kern2", false },
                                               { "File2Kern1", false }};
        for (_pi_offload_entry EntriesIt = EntriesB; EntriesIt != EntriesE; ++EntriesIt) {
            const std::string strFuncname = EntriesIt->name;

            for (const auto& elem: hasKernel) {
                if(strFuncname.find(elem.first) != std::string::npos)
                    hasKernel[elem.first] = true;
            }
        }
        EXPECT_EQ(hasKernel["File1Kern1"], expectedKernel["File1Kern1"]);
        EXPECT_EQ(hasKernel["File1Kern2"], expectedKernel["File1Kern2"]);
        EXPECT_EQ(hasKernel["File2Kern1"], expectedKernel["File2Kern1"]);
    }
    *selected_image_ind = 0;
    return PI_SUCCESS;
}
    
class DeviceCodeSplit : public ::testing::Test {
protected:
    cl::sycl::queue Q;
    sycl::unittest::PiMock Mock;

    DeviceCodeSplit()
      : Q(cl::sycl::cpu_selector()), Mock(Q) {}

    void SetUp() override {
        Mock.redefine<cl::sycl::detail::PiApiKind::piextDeviceSelectBinary>(piextDeviceSelectBinaryRedefine);
    }
};

} // namespace
