// RUN: %{build} -fno-sycl-early-optimizations -o %t.out
// RUN: %{run} %t.out
//
// XFAIL: hip_nvidia

// The test checks that multiple calls to the same template instantiation of a
// group local memory function result in separate allocations, even with device
// code optimizations disabled (the implementation relies on inlining these
// functions regardless of device code optimization settings).

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/group_local_memory.hpp>

#include <cassert>
#include <vector>

constexpr size_t WgSize = 32;
constexpr size_t WgCount = 4;
constexpr size_t Size = WgSize * WgCount;

class KernelA;

using namespace sycl;

int main() {
  queue Q;
  {
    std::vector<int *> VecA(Size, 0);
    std::vector<int *> VecB(Size, 0);
    buffer<int *, 1> BufA{VecA.data(), range<1>(Size)};
    buffer<int *, 1> BufB{VecB.data(), range<1>(Size)};

    Q.submit([&](handler &Cgh) {
      auto AccA = BufA.get_access<access::mode::read_write>(Cgh);
      auto AccB = BufB.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<KernelA>(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), [=](nd_item<1> Item) {
            multi_ptr<int, access::address_space::local_space,
                      sycl::access::decorated::legacy>
                PtrA = ext::oneapi::group_local_memory_for_overwrite<int>(
                    Item.get_group());
            multi_ptr<int, access::address_space::local_space,
                      sycl::access::decorated::legacy>
                PtrB = ext::oneapi::group_local_memory_for_overwrite<int>(
                    Item.get_group());

            size_t GlobalId = Item.get_global_linear_id();
            AccA[GlobalId] = PtrA;
            AccB[GlobalId] = PtrB;
          });
    });

    host_accessor AccA(BufA, read_only);
    host_accessor AccB(BufB, read_only);
    for (size_t I = 0; I < Size; ++I)
      assert(AccA[I] != AccB[I]);
  }
}
