// REQUIRES: aspect-usm_shared_allocations
//
// On CPU it segfaults within the kernel that performs virtual function call.
// https://github.com/intel/llvm/issues/15080
// XFAIL: cpu
// UNSUPPORTED: gpu
// On GPU this test (its older version which used nd_item instead of group)
// used to fail with UR_RESULT_ERROR_PROGRAM_LINK_FAILURE.
// SPIR-V files produced by SYCL_DUMP_IMAGES could be linked just fine (using
// both llvm-spirv -r + llvm-link and ocloc).
// Current version hangs and therefore it is marked as unsupported to avoid
// wasting time in CI and potentially blocking a machine.
// Reported in https://github.com/intel/llvm/issues/15068
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
};

class SumOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int apply(int *LocalData, sycl::group<1> WG) {
    LocalData[WG.get_local_id()] += WG.get_local_id();
    sycl::group_barrier(WG);
    int Res = 0;
    if (WG.leader()) {
      for (size_t I = 0; I < WG.get_local_range().size(); ++I) {
        Res += LocalData[I];
      }
    }

    return sycl::group_broadcast(WG, Res);
  }
};

class MultiplyOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int apply(int *LocalData, sycl::group<1> WG) {
    LocalData[WG.get_local_id()] += WG.get_local_id();
    sycl::group_barrier(WG);
    int Res = 1;
    if (WG.leader()) {
      for (size_t I = 0; I < WG.get_local_range().size(); ++I) {
        Res *= LocalData[I];
      }
    }

    return sycl::group_broadcast(WG, Res);
  }
};

int main() try {
  using storage_t = obj_storage_t<SumOp, MultiplyOp>;

  sycl::queue q;

  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);
  // Let's keep ranges small, or otherwise we will encounter integer overflow
  // (which is a UB) in MultiplyOp::apply.
  sycl::range G{16};
  sycl::range L{4};

  constexpr oneapi::properties props{oneapi::assume_indirect_calls};
  for (unsigned TestCase = 0; TestCase < 2; ++TestCase) {
    std::vector<int> HostData(G.size());
    std::iota(HostData.begin(), HostData.end(), 1);
    std::vector<int> DeviceData = HostData;
    sycl::buffer<int> DataStorage(DeviceData.data(), G);

    q.submit([&](sycl::handler &CGH) {
       CGH.single_task([=]() {
         DeviceStorage->construct</* ret type = */ BaseOp>(TestCase);
       });
     }).wait_and_throw();

    q.submit([&](sycl::handler &CGH) {
      sycl::accessor DataAcc(DataStorage, CGH, sycl::read_write);
      sycl::local_accessor<int> LocalAcc(L, CGH);
      CGH.parallel_for(sycl::nd_range{G, L}, props, [=](auto It) {
        LocalAcc[It.get_local_id()] = DataAcc[It.get_global_id()];
        auto *Ptr = DeviceStorage->getAs<BaseOp>();
        DataAcc[It.get_global_id()] = Ptr->apply(
            LocalAcc.get_multi_ptr<sycl::access::decorated::no>().get(),
            It.get_group());
      });
    });

    // We can't call group_barrier on host and therefore here we have a
    // reference function instead of calling the same methods on host.
    //
    // 'apply' function is written as a kernel, i.e. it describes a single
    // work-item in an nd-range. Here we emulate that nd-range by looping over
    // all work-groups and then over each work-item within that group.
    for (size_t WorkGroupID = 0; WorkGroupID < G.size() / L.size();
         ++WorkGroupID) {
      // Equivalent of a local accessor (LocalData)
      std::vector<int> LocalHostData(L.size());
      // For each work-item within a group, LID - local id
      for (size_t LID = 0; LID < L.size(); ++LID) {
        // GID - global id
        size_t GID = WorkGroupID * L.size() + LID;
        LocalHostData[LID] = HostData[GID];

        // Below (including other loops) is an equivalent of apply's body, but
        // it combains both SumOp and MultiplyOp and hence conditions based on
        // TestCase.
        LocalHostData[LID] += LID;
      }

      // Group barrier is simulated by splitting work-group loop in two.
      // Even though Res is a private variable in the kernel, here we have to
      // declare it in an outer scope (making it local) so it survies our
      // barriers emulation.
      int Res = (TestCase == 0) ? 0 : 1;

      for (size_t LID = 0; LID < L.size(); ++LID) {
        if (LID == 0) { // if that is a group leader
          for (size_t NestedLID = 0; NestedLID < L.size(); ++NestedLID) {
            if (TestCase == 0)
              Res += LocalHostData[NestedLID];
            else
              Res *= LocalHostData[NestedLID];
          }
        }
      }

      // Group broadcast involves a barrier, so we once again splitting
      // work-group loop.
      for (size_t LID = 0; LID < L.size(); ++LID) {
        // GID - global id
        size_t GID = WorkGroupID * L.size() + LID;
        // The broadcast itself: all work-items get result computed by a
        // work-group leader.
        HostData[GID] = Res;
      }
    }

    sycl::host_accessor HostAcc(DataStorage);
    for (size_t I = 0; I < HostData.size(); ++I) {
      if (HostAcc[I] != HostData[I]) {
        std::cout << "Mismatch at index " << I << ": " << HostAcc[I]
                  << " != " << HostData[I] << std::endl;
        assert(HostAcc[I] == HostData[I]);
      }
    }
  }

  sycl::free(DeviceStorage, q);

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
