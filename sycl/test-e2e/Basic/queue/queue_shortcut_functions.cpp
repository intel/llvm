// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

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
#include <vector>

int main() {
  sycl::queue Q;

  constexpr std::size_t Size = 5;

  {
    const int FillValue = 42;
    std::vector<int> Data(Size, 0);

    {
      sycl::buffer<int> DataBuffer(Data.data(), sycl::range<1>(Size));

      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::device,
                     sycl::access::placeholder::true_t,
                     sycl::ext::oneapi::accessor_property_list<>>
          Acc(DataBuffer);

      Q.fill(Acc, FillValue);
      Q.update_host(Acc);
      Q.wait();
    }

    for (int i = 0; i < Size; ++i) {
      if (Data[i] != FillValue) {
        std::cerr << "Incorrect result in fill/update_host test (index " << i
                  << "): got " << Data[i] << ", expected " << FillValue
                  << std::endl;
        return 1;
      }
    }
  }

  {
    const std::vector<int> ReferenceData = {1, 2, 3, 4, 5};
    std::vector<int> Data(Size, 0);
    std::vector<int> CopyBackData(Size, 0);

    {
      sycl::buffer<int> Buf(Data.data(), sycl::range<1>(Size));

      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::device,
                     sycl::access::placeholder::true_t,
                     sycl::ext::oneapi::accessor_property_list<>>
          Acc(Buf);
      Q.copy(ReferenceData.data(), Acc);
      Q.wait();

      Q.copy(Acc, CopyBackData.data());
      Q.wait();
    }

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != Data[i]) {
        std::cerr << "Incorrect result in copy(ptr,acc) at index " << i
                  << ": got " << Data[i] << ", expected " << ReferenceData[i]
                  << std::endl;
        return 1;
      }
    }

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != CopyBackData[i]) {
        std::cerr << "Incorrect result in copy(acc,ptr) at index " << i
                  << ": got " << CopyBackData[i] << ", expected "
                  << ReferenceData[i] << std::endl;
        return 1;
      }
    }
  }

  {
    const std::vector<int> ReferenceData = {2, 4, 6, 8, 10};
    std::shared_ptr<int[]> DataPtr(new int[Size]{0});
    std::shared_ptr<int[]> CopyBackDataPtr(new int[Size]{0});

    {
      sycl::buffer<int> Buf(DataPtr, sycl::range<1>(Size));

      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::device,
                     sycl::access::placeholder::true_t,
                     sycl::ext::oneapi::accessor_property_list<>>
          Acc(Buf);
      Q.copy(ReferenceData.data(), Acc);
      Q.wait();

      Q.copy(Acc, CopyBackDataPtr);
      Q.wait();
    }

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != DataPtr.get()[i]) {
        std::cerr << "Incorrect result in copy(shared_ptr,acc) at index " << i
                  << ": got " << DataPtr.get()[i] << ", expected "
                  << ReferenceData[i] << std::endl;
        return 1;
      }
    }

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != CopyBackDataPtr.get()[i]) {
        std::cerr << "Incorrect result in copy(acc,shared_ptr) at index " << i
                  << ": got " << CopyBackDataPtr.get()[i] << ", expected "
                  << ReferenceData[i] << std::endl;
        return 1;
      }
    }
  }

  {
    const std::vector<int> ReferenceData = {3, 6, 9, 12, 15};
    std::vector<int> Data(Size, 0);

    {
      sycl::buffer<int> RefBuf(ReferenceData.data(), sycl::range<1>(Size));
      sycl::buffer<int> DataBuf(Data.data(), sycl::range<1>(Size));

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
      Q.wait();
    }

    for (int i = 0; i < Size; ++i) {
      if (ReferenceData[i] != Data[i]) {
        std::cerr << "Incorrect result in copy(acc,acc) at index " << i
                  << ": got " << Data[i] << ", expected " << ReferenceData[i]
                  << std::endl;
        return 1;
      }
    }
  }

  return 0;
}
