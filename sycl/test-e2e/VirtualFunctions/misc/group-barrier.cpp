// REQUIRES: aspect-usm_shared_allocations
//
// On CPU it segfaults within the kernel that performs virtual function call.
// XFAIL: cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/15080
// UNSUPPORTED: gpu
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/15068
// On GPU this test (its older version which used nd_item instead of group)
// used to fail with UR_RESULT_ERROR_PROGRAM_LINK_FAILURE.
// SPIR-V files produced by SYCL_DUMP_IMAGES could be linked just fine (using
// both llvm-spirv -r + llvm-link and ocloc).
// Current version hangs and therefore it is marked as unsupported to avoid
// wasting time in CI and potentially blocking a machine.
//
// This test checks that group operations (barrier in this case) work correctly
// inside virtual functions.
//
// RUN: %{build} -o %t.out %helper-includes
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

#include <iostream>
#include <numeric>

namespace oneapi = sycl::ext::oneapi::experimental;

class BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int apply(int *, sycl::group<1>) = 0;

  virtual int computeReference(sycl::range<1> LocalRange, int Init) = 0;
};

class SumOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  int apply(int *LocalData, sycl::group<1> WG) override {
    LocalData[WG.get_local_id()] = WG.get_local_id() + WG.get_group_id();
    sycl::group_barrier(WG);
    if (WG.leader()) {
      int Res = 0;
      for (size_t I = 0; I < WG.get_local_range().size(); ++I) {
        Res += LocalData[I];
      }
      LocalData[0] = Res;
    }
    sycl::group_barrier(WG);

    return LocalData[0];
  }

  int computeReference(sycl::range<1> LocalRange, int WGID) override {
    std::vector<int> LocalData(LocalRange.size());
    for (size_t LID = 0; LID < LocalRange.size(); ++LID)
      LocalData[LID] = LID + WGID;

    int Res = 0;
    for (size_t LID = 0; LID < LocalRange.size(); ++LID)
      Res += LocalData[LID];

    return Res;
  }
};

class MultiplyOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  int apply(int *LocalData, sycl::group<1> WG) override {
    // +1 to avoid multiplying by 0 below
    LocalData[WG.get_local_id()] = WG.get_local_id() + WG.get_group_id() + 1;
    sycl::group_barrier(WG);
    if (WG.leader()) {
      int Res = 1;
      for (size_t I = 0; I < WG.get_local_range().size(); ++I) {
        Res *= LocalData[I];
      }
      LocalData[0] = Res;
    }
    sycl::group_barrier(WG);

    return LocalData[0];
  }

  int computeReference(sycl::range<1> LocalRange, int WGID) override {
    std::vector<int> LocalData(LocalRange.size());
    for (size_t LID = 0; LID < LocalRange.size(); ++LID)
      LocalData[LID] = LID + WGID + 1;

    int Res = 1;
    for (size_t LID = 0; LID < LocalRange.size(); ++LID)
      Res *= LocalData[LID];

    return Res;
  }
};

int main() try {
  using storage_t = obj_storage_t<SumOp, MultiplyOp>;

  sycl::queue q;

  storage_t HostStorage;
  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);
  // Let's keep ranges small, or otherwise we will encounter integer overflow
  // (which is a UB) in MultiplyOp::apply.
  sycl::range G{16};
  sycl::range L{4};

  constexpr oneapi::properties props{oneapi::assume_indirect_calls};
  for (unsigned TestCase = 0; TestCase < 2; ++TestCase) {
    sycl::buffer<int> DataStorage(G);

    q.submit([&](sycl::handler &CGH) {
       CGH.single_task([=]() {
         DeviceStorage->construct</* ret type = */ BaseOp>(TestCase);
       });
     }).wait_and_throw();

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor DataAcc(DataStorage, CGH, sycl::read_write);
      sycl::local_accessor<int> LocalAcc(L, CGH);
      CGH.parallel_for(sycl::nd_range{G, L}, props, [=](auto It) {
        auto *Ptr = DeviceStorage->getAs<BaseOp>();
        DataAcc[It.get_global_id()] = Ptr->apply(
            LocalAcc.get_multi_ptr<sycl::access::decorated::no>().get(),
            It.get_group());
      });
    }).wait_and_throw();

    auto *Ptr = HostStorage.construct</* ret type = */ BaseOp>(TestCase);
    sycl::host_accessor HostAcc(DataStorage);

    // All work-items in a group produce the same result, so we do verification
    // per work-group.
    for (size_t WorkGroupID = 0; WorkGroupID < G.size() / L.size();
         ++WorkGroupID) {
      int Reference = Ptr->computeReference(L, WorkGroupID);
      for (size_t I = 0; I < L.size(); ++I) {
        size_t GID = WorkGroupID * L.size() + I;
        if (HostAcc[GID] != Reference) {
          std::cout << "Mismatch at index " << I << ": " << HostAcc[I]
                    << " != " << Reference << std::endl;
          assert(HostAcc[I] == Reference);
        }
      }
    }
  }

  sycl::free(DeviceStorage, q);

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
