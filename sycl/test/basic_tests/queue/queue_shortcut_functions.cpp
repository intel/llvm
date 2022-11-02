// RUN: %clangxx -fsycl -fsyntax-only %s -o %t.out

//==-- queue_shortcut_functions.cpp - SYCL queue shortcut functions test ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//==------------------------------------------------------------------------==//

#include <sycl/sycl.hpp>

#include <iostream>
#include <memory>

int main() {
  sycl::queue Q;

  constexpr int Size = 5;

  {
    const int FillValue = 42;
    int Data[Size] = {0};
    sycl::buffer<int> DataBuffer(Data, Size);

    sycl::accessor<int, 1, sycl::access::mode::write,
                   sycl::access::target::device,
                   sycl::access::placeholder::true_t,
                   sycl::ext::oneapi::accessor_property_list<>>
        DataAccWrite(DataBuffer);

    Q.fill(DataAccWrite, FillValue);

    sycl::accessor<int, 1, sycl::access::mode::read,
                   sycl::access::target::device,
                   sycl::access::placeholder::true_t,
                   sycl::ext::oneapi::accessor_property_list<>>
        DataAccRead(DataBuffer);

    Q.update_host(DataAccRead);

    for (int i = 0; i < Size; ++i) {
      if (Data[i] != FillValue) {
        std::cerr << "Incorrect result in fill/update_host test: got "
                  << Data[i] << ", expected " << FillValue << std::endl;
        return 1;
      }
    }
  }

  {
    int ReferenceData[Size] = {1, 2, 3, 4, 5};
    int Data[Size] = {0};
    int CopyBackData[Size] = {0};

    sycl::buffer<int> Buf(Data, Size);

    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::device,
                   sycl::access::placeholder::true_t,
                   sycl::ext::oneapi::accessor_property_list<>>
        Acc(Buf);
    Q.copy(ReferenceData, Acc);

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != Data[i]) {
        std::cerr << "Incorrect result at index " << i << ": got " << Data[i]
                  << ", expected " << ReferenceData[i] << std::endl;
        return 1;
      }
    }

    Q.copy(Acc, CopyBackData);

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != CopyBackData[i]) {
        std::cerr << "Incorrect result at index " << i << ": got "
                  << CopyBackData[i] << ", expected " << ReferenceData[i]
                  << std::endl;
        return 1;
      }
    }
  }

  {
    int ReferenceData[Size] = {1, 2, 3, 4, 5};
    std::shared_ptr<int[]> DataPtr = std::make_shared<int[]>();
    std::shared_ptr<int[]> CopyBackDataPtr = std::make_shared<int[]>();

    sycl::buffer<int> Buf(DataPtr, Size);

    sycl::accessor<int, 1, sycl::access::mode::read_write,
                   sycl::access::target::device,
                   sycl::access::placeholder::true_t,
                   sycl::ext::oneapi::accessor_property_list<>>
        Acc(Buf);
    Q.copy(ReferenceData, Acc);

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != DataPtr[i]) {
        std::cerr << "Incorrect result at index " << i << ": got " << DataPtr[i]
                  << ", expected " << ReferenceData[i] << std::endl;
        return 1;
      }
    }

    Q.copy(Acc, CopyBackDataPtr);

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != CopyBackDataPtr[i]) {
        std::cerr << "Incorrect result at index " << i << ": got "
                  << CopyBackDataPtr[i] << ", expected " << ReferenceData[i]
                  << std::endl;
        return 1;
      }
    }
  }

  {
    int ReferenceData[Size] = {1, 2, 3, 4, 5};
    int Data[Size] = {0};

    sycl::buffer<int> RefBuf(ReferenceData, Size);
    sycl::buffer<int> DataBuf(Data, Size);

    sycl::accessor<int, 1, sycl::access::mode::read,
                   sycl::access::target::device,
                   sycl::access::placeholder::true_t,
                   sycl::ext::oneapi::accessor_property_list<>>
        RefAcc(RefBuf);
    sycl::accessor<int, 1, sycl::access::mode::write,
                   sycl::access::target::device,
                   sycl::access::placeholder::true_t,
                   sycl::ext::oneapi::accessor_property_list<>>
        DataAcc(DataBuf);

    Q.copy(RefAcc, DataAcc);

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != Data[i]) {
        std::cerr << "Incorrect result at index " << i << ": got " << Data[i]
                  << ", expected " << ReferenceData[i] << std::endl;
        return 1;
      }
    }
  }

  return 0;
}