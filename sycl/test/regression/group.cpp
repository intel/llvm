// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

//==-- group.cpp - Regression tests for cl::sycl::group API bug fixes. -----==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

struct AsyncHandler {
  void operator()(cl::sycl::exception_list L) {
    for (auto &E : L) {
      std::rethrow_exception(E);
    }
  }
};

// Tests group::get_group_range
bool group__get_group_range() {
  std::cout << "+++ Running group::get_group_range() test...\n";
  constexpr int DIMS = 3;
  const range<DIMS> LocalRange{2, 3, 1};
  const range<DIMS> GroupRange{1, 2, 3};
  const range<DIMS> GlobalRange = LocalRange * GroupRange;
  using DataType = size_t;
  const int DataLen = GlobalRange.size() * DIMS;
  std::unique_ptr<DataType[]> Data(new DataType[DataLen]);
  std::memset(Data.get(), 0, DataLen * sizeof(DataType));

  try {
    buffer<DataType, 1> Buf(Data.get(), DataLen);
    queue Q(AsyncHandler{});

    Q.submit([&](handler &cgh) {
      auto Ptr = Buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class group__get_group_range>(
          nd_range<3>{GlobalRange, LocalRange}, [=](nd_item<DIMS> I) {
            const auto Off = I.get_global_linear_id() * 3;
            const auto &GR = I.get_group().get_group_range();
            Ptr[Off + 0] = GR.get(0);
            Ptr[Off + 1] = GR.get(1);
            Ptr[Off + 2] = GR.get(2);
          });
    });
  } catch (cl::sycl::exception const &E) {
    std::cout << "SYCL exception caught: " << E.what() << '\n';
    return 2;
  }
  const size_t SIZE_Z = GlobalRange.get(0);
  const size_t SIZE_Y = GlobalRange.get(1);
  const size_t SIZE_X = GlobalRange.get(2);
  bool Pass = true;
  int ErrCnt = 0;

  for (size_t Z = 0; Z < SIZE_Z; Z++) {
    for (size_t Y = 0; Y < SIZE_Y; Y++) {
      for (size_t X = 0; X < SIZE_X; X++) {
        const size_t Ind = Z * SIZE_X * SIZE_Y + Y * SIZE_X + X;
        const size_t Off = Ind * 3;
        const auto Ptr = Data.get();
        const auto XTest = Ptr[Off + 2];
        const auto YTest = Ptr[Off + 1];
        const auto ZTest = Ptr[Off + 0];
        const auto XGold = GroupRange.get(2);
        const auto YGold = GroupRange.get(1);
        const auto ZGold = GroupRange.get(0);
        const bool Ok = (XTest == XGold && YTest == YGold && ZTest == ZGold);
        Pass &= Ok;

        if (!Ok && ErrCnt++ < 10) {
          std::cout << "*** ERROR at [" << Z << "][" << Y << "][" << X << "]: ";
          std::cout << XTest << " " << YTest << " " << ZTest << " != ";
          std::cout << XGold << " " << YGold << " " << ZGold << "\n";
        }
      }
    }
  }
  if (Pass)
    std::cout << "    pass\n";
  return Pass;
}

// Tests group::get_linear_id()
bool group__get_linear_id() {
  std::cout << "+++ Running group::get_linear_id() test...\n";
  constexpr int DIMS = 3;
  const range<DIMS> LocalRange{2, 3, 1};
  const range<DIMS> GroupRange{1, 2, 3};
  const range<DIMS> GlobalRange = LocalRange * GroupRange;
  using DataType = size_t;
  const int DataLen = GlobalRange.size() * DIMS;
  std::unique_ptr<DataType[]> Data(new DataType[DataLen]);
  std::memset(Data.get(), 0, DataLen * sizeof(DataType));

  try {
    buffer<DataType, 1> Buf(Data.get(), DataLen);
    queue Q(AsyncHandler{});

    Q.submit([&](handler &cgh) {
      auto Ptr = Buf.get_access<access::mode::write>(cgh);

      cgh.parallel_for<class group__get_linear_id>(
          nd_range<3>{GlobalRange, LocalRange}, [=](nd_item<DIMS> I) {
            const auto Off = I.get_global_linear_id() * 3;
            const auto LI = I.get_group().get_linear_id();
            Ptr[Off + 0] = LI;
            Ptr[Off + 1] = LI;
            Ptr[Off + 2] = LI;
          });
    });
  } catch (cl::sycl::exception const &E) {
    std::cout << "SYCL exception caught: " << E.what() << '\n';
    return 2;
  }
  const size_t SIZE_Z = GlobalRange.get(0);
  const size_t SIZE_Y = GlobalRange.get(1);
  const size_t SIZE_X = GlobalRange.get(2);
  bool Pass = true;
  int ErrCnt = 0;

  for (size_t Z = 0; Z < SIZE_Z; Z++) {
    for (size_t Y = 0; Y < SIZE_Y; Y++) {
      for (size_t X = 0; X < SIZE_X; X++) {
        const size_t Ind = Z * SIZE_X * SIZE_Y + Y * SIZE_X + X;
        const size_t Off = Ind * 3;
        const auto Ptr = Data.get();
        const auto XTest = Ptr[Off + 2];
        const auto YTest = Ptr[Off + 1];
        const auto ZTest = Ptr[Off + 0];
        const id<3> GlobalID{Z, Y, X};
        const id<3> GroupID = GlobalID / LocalRange;
        const auto GroupLinearID =
            GroupID.get(0) * GroupRange.get(1) * GroupRange.get(2) +
            GroupID.get(1) * GroupRange.get(2) + GroupID.get(2);
        const auto XGold = GroupLinearID;
        const auto YGold = GroupLinearID;
        const auto ZGold = GroupLinearID;
        const bool Ok = (XTest == XGold && YTest == YGold && ZTest == ZGold);
        Pass &= Ok;

        if (!Ok && ErrCnt++ < 10) {
          std::cout << "*** ERROR at [" << Z << "][" << Y << "][" << X << "]: ";
          std::cout << XTest << " " << YTest << " " << ZTest << " != ";
          std::cout << XGold << " " << YGold << " " << ZGold << "\n";
        }
      }
    }
  }
  if (Pass)
    std::cout << "    pass\n";
  return Pass;
}

int main() {
  bool Pass = 1;
  Pass &= group__get_group_range();
  Pass &= group__get_linear_id();

  if (!Pass) {
    std::cout << "FAILED\n";
    return 1;
  }
  std::cout << "Passed\n";
  return 0;
}
