// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// XFAIL: hip_nvidia

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/group_local_memory.hpp>

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

int main() {
  queue Q;
  {
    std::vector<int> Vec(Size, 0);
    buffer<int, 1> Buf{Vec.data(), range<1>(Size)};
    std::vector<int> CounterVec(WgCount, 0);
    buffer<int, 1> CounterBuf{CounterVec.data(), range<1>(WgCount)};

    Q.submit([&](handler &Cgh) {
      auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
      auto CounterAcc = CounterBuf.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<KernelA>(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), [=](nd_item<1> Item) {
            // Some alternative (and functionally equivalent) ways to use this
            // would be:
            // auto Ptr = group_local_memory<Foo>(Item.get_group(), ...);
            // Foo &Ref = *group_local_memory<Foo>(Item.get_group(), ...);
            multi_ptr<Foo, access::address_space::local_space,
                      sycl::access::decorated::legacy>
                Ptr = sycl::ext::oneapi::group_local_memory<Foo>(
                    Item.get_group(), 1,
                    CounterAcc[Item.get_group_linear_id()]);
            Ptr->Values[Item.get_local_linear_id()] *=
                Item.get_local_linear_id();

            Item.barrier();
            // Check that the memory is accessible from other work-items
            size_t LocalIdx = Item.get_local_linear_id() ^ 1;
            size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
            Acc[GlobalIdx] = Ptr->Values[LocalIdx];
          });
    });

    host_accessor Acc(Buf, read_only);
    for (size_t I = 0; I < Size; ++I)
      assert(Acc[I] == I % WgSize);

    // Check that the constructor has been called once per work-group
    host_accessor CounterAcc(CounterBuf, read_only);
    for (size_t I = 0; I < WgCount; ++I)
      assert(CounterAcc[I] == 1);
  }

  {
    std::vector<int> Vec(Size, 0);
    buffer<int, 1> Buf{Vec.data(), range<1>(Size)};

    Q.submit([&](handler &Cgh) {
      auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
      Cgh.parallel_for<KernelB>(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), [=](nd_item<1> Item) {
            multi_ptr<int[WgSize], access::address_space::local_space,
                      sycl::access::decorated::legacy>
                Ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<
                    int[WgSize]>(Item.get_group());
            (*Ptr)[Item.get_local_linear_id()] = Item.get_local_linear_id();

            Item.barrier();
            // Check that the memory is accessible from other work-items
            size_t LocalIdx = Item.get_local_linear_id() ^ 1;
            size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
            Acc[GlobalIdx] = (*Ptr)[LocalIdx];
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
      auto Acc = Buf.get_access<access::mode::write>(Cgh);
      Cgh.parallel_for<KernelC>(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), [=](nd_item<1> Item) {
            multi_ptr<Bar, access::address_space::local_space,
                      sycl::access::decorated::legacy>
                Ptr = sycl::ext::oneapi::group_local_memory_for_overwrite<Bar>(
                    Item.get_group());
            // Make sure that the value is default initialized on all items.
            Acc[Item.get_global_linear_id()] = Ptr->Value;
          });
    });

    host_accessor Acc(Buf, read_only);
    Bar RefBar;
    for (size_t I = 0; I < Size; ++I)
      assert(Acc[I] == RefBar.Value);
  }
}
