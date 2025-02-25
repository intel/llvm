// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/work_group_static.hpp>

#include <cassert>
#include <vector>

constexpr size_t WgSize = 32;
constexpr size_t WgCount = 4;
constexpr size_t Size = WgSize * WgCount;

struct Foo {
  Foo() = delete;
  Foo(int Value, int &Counter) {
    for (int I = 0; I < WgSize; ++I)
      Values[I] = Value;
    ++Counter;
  }
  int Values[WgSize];
};

struct Bar {
  int Value = 42;
};

class KernelA;
class KernelB;
class KernelC;

using namespace sycl;

struct LocalMem {
  // Local mem used in kernel
  static sycl::ext::oneapi::experimental::work_group_static<int[WgSize]>
      localIDBuff;
};
sycl::ext::oneapi::experimental::work_group_static<int[WgSize]>
    LocalMem::localIDBuff;

int main() {
  queue Q;

  {
    std::vector<int> Vec(Size, 0);
    buffer<int, 1> Buf{Vec.data(), range<1>(Size)};

    Q.submit([&](handler &Cgh) {
      auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<KernelA>(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), [=](nd_item<1> Item) {
            LocalMem::localIDBuff[Item.get_local_linear_id()] =
                Item.get_local_linear_id();

            Item.barrier();
            // Check that the memory is accessible from other work-items
            size_t LocalIdx = Item.get_local_linear_id() ^ 1;
            size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
            Acc[GlobalIdx] = LocalMem::localIDBuff[LocalIdx];
          });
    });

    host_accessor Acc(Buf, read_only);
    for (size_t I = 0; I < Size; ++I)
      assert(Acc[I] == I % WgSize);
  }

  {
    std::vector<int> Vec(Size, 0);
    buffer<int, 1> Buf{Vec.data(), range<1>(Size)};

    Q.submit([&](handler &Cgh) {
      auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<KernelB>(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), [=](nd_item<1> Item) {
            sycl::ext::oneapi::experimental::work_group_static<int> localIDBuff;
            int id = Item.get_global_linear_id();
            if (Item.get_group().leader())
              localIDBuff = id;

            Item.barrier();
            // Check that the memory is accessible from other work-items
            size_t GlobalIdx = Item.get_global_linear_id();
            Acc[GlobalIdx] = localIDBuff;
          });
    });

    host_accessor Acc(Buf, read_only);
    for (size_t I = 0; I < Size; ++I)
      assert(Acc[I] == (I / WgSize) * WgSize);
  }
}
