// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %RUN_ON_HOST %t.out

//==-struct_kernel_param.cpp-Checks passing structs as kernel params--------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>

using namespace cl::sycl;

struct MyNestedStruct {
  bool operator==(const MyNestedStruct &Rhs) {
    return (FldArr[0] == Rhs.FldArr[0] && FldFloat == Rhs.FldFloat);
  }
  cl::sycl::cl_char FldArr[1];
  cl::sycl::cl_float FldFloat;
};

struct MyStruct {
  bool operator==(const MyStruct &Rhs) {
    return (FldChar == Rhs.FldChar && FldLong == Rhs.FldLong &&
            FldShort == Rhs.FldShort && FldUint == Rhs.FldUint &&
            FldStruct == Rhs.FldStruct &&
            std::equal(std::begin(FldArr), std::end(FldArr),
                       std::begin(Rhs.FldArr)) &&
            FldInt == Rhs.FldInt);
  }
  cl::sycl::cl_char FldChar;
  cl::sycl::cl_long FldLong;
  cl::sycl::cl_short FldShort;
  cl::sycl::cl_uint FldUint;
  MyNestedStruct FldStruct;
  cl::sycl::cl_short FldArr[3];
  cl::sycl::cl_int FldInt;
};

MyStruct GlobS;

static void printStruct(const MyStruct &S0) {
  std::cout << "{ " << (int)S0.FldChar << ", " << S0.FldLong << ", "
            << S0.FldShort << ", " << S0.FldUint << " { { "
            << (int)S0.FldStruct.FldArr[0] << " }, " << S0.FldStruct.FldFloat
            << " }, { " << S0.FldArr[0] << ", " << S0.FldArr[1] << ", "
            << S0.FldArr[2] << " }, " << S0.FldInt << " }";
}

bool test0() {
  MyStruct S = GlobS;
  MyStruct S0 = {0};
  {
    buffer<MyStruct, 1> Buf(&S0, range<1>(1));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buf.get_access<access::mode::write>(cgh);
      cgh.single_task<class MyKernel>([=] { B[0] = S; });
    });
  }
  bool Passed = (S == S0);

  if (!Passed) {
    std::cout << "test0 failed" << std::endl;
    std::cout << "test0 input:" << std::endl;
    printStruct(S);
    std::cout << std::endl;
    std::cout << "test0 result:\n";
    printStruct(S0);
    std::cout << std::endl;
  }
  return Passed;
}

bool test1() {
  range<3> ice(8, 9, 10);
  uint ice2 = 888;
  uint result[4] = {0};

  {
    buffer<unsigned int, 1> Buffer((unsigned int *)result, range<1>(4));
    queue myQueue;
    myQueue.submit([&](handler &cgh) {
      auto B = Buffer.get_access<access::mode::write>(cgh);
      cgh.parallel_for<class bufferByRange_cap>(range<1>{4}, [=](id<1> index) {
        B[index.get(0)] = index.get(0) > 2 ? ice2 : ice.get(index.get(0));
      });
    });
  }

  bool Passed = true;

  for (unsigned long i = 0; i < 4; ++i) {
    if (i <= 2) {
      if (result[i] != ice[i])
        Passed = false;
    } else {
      if (result[i] != ice2)
        Passed = false;
    }
  }
  if (!Passed)
    std::cout << "test1 failed" << std::endl;

  return Passed;
}

int main(int argc, char **argv) {
  cl::sycl::cl_char PartChar = argc;
  cl::sycl::cl_short PartShort = argc << 8;
  cl::sycl::cl_int PartInt = argc << 16;
  cl::sycl::cl_uint PartUint = argc << 16;
  cl::sycl::cl_long PartLong = ((cl::sycl::cl_long)argc) << 32;
  cl::sycl::cl_float PartFloat = argc;

  GlobS = {PartChar,
           PartLong,
           PartShort,
           PartUint,
           {{PartChar}, PartFloat},
           {PartShort, PartShort, PartShort},
           PartInt};

  bool Pass = test0() & test1();

  std::cout << "Test " << (Pass ? "passed" : "FAILED") << std::endl;
  return Pass ? 0 : 1;
}
