//==---------- device-code-split-lib.hpp --- Device code split unit tests --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <CL/sycl.hpp>

class File1Kern1;
class File1Kern2;
class File2Kern1;

extern "C" __SYCL_EXPORT void runKernel1FromFile1(cl::sycl::queue Q, cl::sycl::buffer<int, 1> Buf);
extern "C" __SYCL_EXPORT void runKernel2FromFile1(cl::sycl::queue Q, cl::sycl::buffer<int, 1> Buf);
extern "C" __SYCL_EXPORT void runKernel1FromFile2(cl::sycl::queue Q, cl::sycl::buffer<int, 1> Buf);
