// REQUIRES: native_cpu_ock

// Tests that no linker errors occur when group collective functions are used
// in conjuction with SYCL_EXTERNAL.

// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DFILE1 -c -o %t1.o %s
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu -DFILE2 -c -o %t2.o %s
// RUN: llvm-ar crv %t1.a %t1.o
// RUN: %clangxx -fsycl -fsycl-targets=native_cpu %t2.o %t1.a -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=native_cpu:cpu %t.out

/*
    test performs a lattice reduction.
    sycl::vec<float> is sensitive to .get_size() vs .size() in SYCL headers
    (ie, byte size versus vector size)
*/
#include <cmath>
#include <cstdio>
#include <iostream>

#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/usm.hpp>

using namespace sycl;

constexpr int ItemsPerGroup = 32;
constexpr int NumGroups = 2;
constexpr int VecWidth = 8;
constexpr int N = ItemsPerGroup * NumGroups * VecWidth;
constexpr double Tolerance = 1e-10;
using vecn = sycl::vec<float, VecWidth>;
#ifdef FILE1

SYCL_EXTERNAL void groupSum(vecn *GroupSums, const vecn &InVec, const int GrId,
                            sycl::group<2> &Group, const int Zid) {

  vecn Contribution = (Zid == GrId ? InVec : vecn(0));
  auto Sum = reduce_over_group(Group, Contribution, sycl::plus<>());
  if (Zid == GrId && Group.get_local_id()[1] == 0)
    GroupSums[GrId] = Sum;
}
#endif

#ifdef FILE2
SYCL_EXTERNAL void groupSum(vecn *GroupSums, const vecn &InVec, const int GrId,
                            sycl::group<2> &Group, const int Zid);
void test(queue q, float *GroupSums, float *InData) {

  sycl::range<2> globalSize(NumGroups, ItemsPerGroup);
  sycl::range<2> localSize(1, ItemsPerGroup);
  sycl::nd_range<2> range{globalSize, localSize};

  q.submit([&](sycl::handler &h) {
    h.parallel_for<>(range, [=](sycl::nd_item<2> ndi) {
      int ItId = ndi.get_global_id(1);
      int GrId = ndi.get_global_id(0);

      auto InVecs = reinterpret_cast<vecn *>(InData);
      auto GroupSumVecs = reinterpret_cast<vecn *>(GroupSums);

      auto ThisGr = ndi.get_group();

      for (int iz = 0; iz < NumGroups; iz++) {
        groupSum(GroupSumVecs, InVecs[GrId * ItemsPerGroup + ItId], GrId,
                 ThisGr, iz);
      }
    });
  });
  q.wait();
}

int main() {

  queue q{default_selector_v};
  auto dev = q.get_device();
  std::cout << "Device: " << dev.get_info<info::device::name>() << std::endl;

  auto ctx = q.get_context();
  auto *InData = (float *)sycl::aligned_alloc_shared(
      alignof(vecn), N * sizeof(float), dev, ctx);
  auto *GroupSums = (float *)sycl::aligned_alloc_shared(
      alignof(vecn), NumGroups * VecWidth * sizeof(float), dev, ctx);

  for (int DataId = 0; DataId < N; DataId++) {
    InData[DataId] = DataId;
  }

  test(q, GroupSums, InData);

  int fails = 0;
  for (int GrId = 0; GrId < NumGroups; GrId++) {
    float ExpectedVecSum[VecWidth] = {0};
    for (int ItId = 0; ItId < ItemsPerGroup; ItId++) {
      for (int VId = 0; VId < VecWidth; VId++) {
        ExpectedVecSum[VId] +=
            InData[(GrId * ItemsPerGroup + ItId) * VecWidth + VId];
      }
    }
    for (int VId = 0; VId < VecWidth; VId++) {
      auto d = ExpectedVecSum[VId] - GroupSums[GrId * VecWidth + VId];
      if (std::abs(d) > Tolerance) {
        std::printf("partial fail ");
        std::printf("%i\t%i\t%g\t%g\n", GrId, VId, ExpectedVecSum[VId],
                    GroupSums[GrId * VecWidth + VId]);
        fails++;
      } else {
        std::printf("partial pass ");
        std::printf("%i\t%i\t%g\t%g\n", GrId, VId, ExpectedVecSum[VId],
                    GroupSums[GrId * VecWidth + VId]);
      }
    }
  }

  if (fails == 0) {
    std::printf("test passed!\n");
  } else {
    std::printf("test failed!\n");
  }
  free(InData, ctx);
  free(GroupSums, ctx);
  return fails;
}
#endif
