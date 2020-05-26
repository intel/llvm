//==----------- split-per-source.cpp --- Device code split unit tests ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "device-code-split-lib.hpp"
#include "device-code-split-utils.hpp"

namespace DeviceCodeSplitTests {

TEST_F(DeviceCodeSplit, SplitPerSourceCheckFile1) {

  expectedKernel = std::map<std::string, bool> {{"File1Kern1", true },
                                                {"File1Kern2", true },
                                                {"File2Kern1", false }};
  int Data = 0;
  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    runKernel1FromFile1(Q, Buf);
  }
  EXPECT_EQ(Data, 1);

  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    runKernel2FromFile1(Q, Buf);
  }
  EXPECT_EQ(Data, 2);
}

TEST_F(DeviceCodeSplit, SplitPerSourceCheckFile2) {

  expectedKernel = std::map<std::string, bool> {{"File1Kern1", false },
                                                {"File1Kern2", false },
                                                {"File2Kern1", true }};
  int Data = 0;
  {
    cl::sycl::buffer<int, 1> Buf(&Data, cl::sycl::range<1>(1));
    runKernel1FromFile2(Q, Buf);
  }
  EXPECT_EQ(Data, 3);
}

} // namespace
