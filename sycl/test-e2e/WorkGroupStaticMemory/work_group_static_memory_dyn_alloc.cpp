// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// REQUIRES: cuda

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


using namespace sycl;

int main() {
  queue Q;
  {
    std::vector<int> Vec(Size, 0);
    buffer<int, 1> Buf{Vec.data(), range<1>(Size)};

    Q.submit([&](handler &Cgh) {
      auto Acc = Buf.get_access<access::mode::read_write>(Cgh);
      sycl::ext::oneapi::experimental::work_group_static_size static_size(
          WgSize * sizeof(int));
      sycl::ext::oneapi::experimental::properties properties{static_size};
      Cgh.parallel_for(nd_range<1>(range<1>(Size), range<1>(WgSize)),
                       properties, [=](nd_item<1> Item) {
                         multi_ptr<int, access::address_space::local_space,
                                   sycl::access::decorated::no>
                             Ptr = sycl::ext::oneapi::experimental::
                                 get_dynamic_work_group_memory<int>();
                         Ptr[Item.get_local_linear_id()] =
                             Item.get_local_linear_id();

                         Item.barrier();
                         // Check that the memory is accessible from other
                         // work-items
                         size_t LocalIdx = Item.get_local_linear_id() ^ 1;
                         size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
                         Acc[GlobalIdx] = Ptr[LocalIdx];
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
      sycl::ext::oneapi::experimental::work_group_static_size static_size(
          WgSize * sizeof(int));
      sycl::ext::oneapi::experimental::properties properties{static_size};
      Cgh.parallel_for(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), properties,
          [=](nd_item<1> Item) {
            multi_ptr<int, access::address_space::local_space,
                      sycl::access::decorated::no>
                Ptr = sycl::ext::oneapi::experimental::
                    get_dynamic_work_group_memory<int>();
            Ptr[Item.get_local_linear_id()] = Item.get_local_linear_id();

            Item.barrier();
            // Check multiple calls returns the same pointer
            multi_ptr<unsigned int, access::address_space::local_space,
                      sycl::access::decorated::no>
                PtrAlias = sycl::ext::oneapi::experimental::
                    get_dynamic_work_group_memory<unsigned int>();
            // Check that the memory is accessible from other work-items
            size_t LocalIdx = Item.get_local_linear_id() ^ 1;
            size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
            Acc[GlobalIdx] = PtrAlias[LocalIdx];
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
      sycl::ext::oneapi::experimental::work_group_static_size static_size(
          WgSize * sizeof(int));
      sycl::ext::oneapi::experimental::properties properties{static_size};
      auto LocalAccessor = sycl::local_accessor<int>(WgSize * sizeof(int), Cgh);
      Cgh.parallel_for(
          nd_range<1>(range<1>(Size), range<1>(WgSize)), properties,
          [=](nd_item<1> Item) {
            multi_ptr<int, access::address_space::local_space,
                      sycl::access::decorated::no>
                Ptr = sycl::ext::oneapi::experimental::
                    get_dynamic_work_group_memory<int>();
            Ptr[Item.get_local_linear_id()] = Item.get_local_linear_id();

            Item.barrier();
            // Check that the memory is accessible from other work-items
            size_t LocalIdx = Item.get_local_linear_id() ^ 1;
            LocalAccessor[Item.get_local_linear_id()] = Ptr[LocalIdx] + 1;
            Item.barrier();

            size_t GlobalIdx = Item.get_global_linear_id() ^ 1;
            Acc[GlobalIdx] = LocalAccessor[Item.get_local_linear_id()];
          });
    });

    host_accessor Acc(Buf, read_only);
    for (size_t I = 0; I < Size; ++I)
      assert(Acc[I] == I % WgSize + 1);
  }
}
