// FIXME: replace unsupported with an aspect check once we have it
// UNSUPPORTED: cuda, hip, acc
//
// REQUIRES: aspect-usm_shared_allocations
//
// Fails with UR_RESULT_ERROR_PROGRAM_LINK_FAILURE. SPIR-V files produced by
// SYCL_DUMP_IMAGES can be linked just fine (using llvm-spirv -r + llvm-link),
// so it seems to be a problem on IGC side.
// Reported in https://github.com/intel/llvm/issues/15068
// On CPU it segfaults within the kernel that performs virtual function call.
// https://github.com/intel/llvm/issues/15080
// XFAIL: gpu, cpu
//
// This test checks that group operations (barrier in this case) work correctly
// inside virtual functions.
//
// RUN: %{build} -o %t.out %helper-includes
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/group_barrier.hpp>
#include <sycl/group_algorithm.hpp>
#include <sycl/usm.hpp>

#include "helpers.hpp"

#include <iostream>
#include <numeric>

namespace oneapi = sycl::ext::oneapi::experimental;

class BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int apply(int *, sycl::nd_item<1>) = 0;
};

class SumOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int apply(int *LocalData, sycl::nd_item<1> It) {
    LocalData[It.get_local_id()] += It.get_local_id();
    sycl::group_barrier(It.get_group());
    int Res = 0;
    if (It.get_group().leader()) {
      for (size_t I = 0; I < It.get_local_range().size(); ++I) {
        Res += LocalData[I];
      }
    }

    return sycl::group_broadcast(It.get_group(), Res);
  }
};

class MultiplyOp : public BaseOp {
public:
  SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(oneapi::indirectly_callable)
  virtual int apply(int *LocalData, sycl::nd_item<1> It) {
    LocalData[It.get_local_id()] += It.get_local_id();
    sycl::group_barrier(It.get_group());
    int Res = 1;
    if (It.get_group().leader()) {
      for (size_t I = 0; I < It.get_local_range().size(); ++I) {
        Res *= LocalData[I];
      }
    }

    return sycl::group_broadcast(It.get_group(), Res);
  }
};

int main() try {
  using storage_t = obj_storage_t<SumOp, MultiplyOp>;

  auto asyncHandler = [](sycl::exception_list list) {
    for (auto &e : list)
      std::rethrow_exception(e);
  };

  sycl::queue q(asyncHandler);

  auto *DeviceStorage = sycl::malloc_shared<storage_t>(1, q);
  sycl::range G{512};
  sycl::range L{32};

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
            LocalAcc.template get_multi_ptr<sycl::access::decorated::no>()
                .get(),
            It);
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

        // Below is an equivalent of apply's body, but it combains both SumOp
        // and MultiplyOp and hence conditions based on TestCase.
        LocalHostData[LID] = LID;

        // group barrier which is no-op here

        int Res = (TestCase == 0) ? 0 : 1;
        if (LID == 0) { // if that is a group leader
          for (size_t NestedLID = 0; NestedLID < L.size(); ++NestedLID) {
            if (TestCase == 0)
              Res += LocalHostData[NestedLID];
            else
              Res *= LocalHostData[NestedLID];
          }
        }

        // group broadcast:
        for (size_t LID = 0; LID < L.size(); ++LID)
          HostData[GID] = Res;
      }
    }

    sycl::host_accessor HostAcc(DataStorage);
    for (size_t I = 0; I < HostData.size(); ++I)
      assert(HostAcc[I] == HostData[I]);
  }

  sycl::free(DeviceStorage, q);

  return 0;
} catch (sycl::exception &e) {
  std::cout << "Unexpected exception was thrown: " << e.what() << std::endl;
  return 1;
}
