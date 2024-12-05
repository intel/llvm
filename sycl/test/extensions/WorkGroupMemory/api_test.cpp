// RUN: %clangxx -fsycl -fsyntax-only %s
#include <cassert>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl;
namespace syclexp = sycl::ext::oneapi::experimental;

queue Q;

struct S {
  int a;
  char b;
};

union U {
  int a;
  char b;
};

template <typename DataT> void convertToDataT(DataT &data) {}

template <typename DataT> void test_constness() {
  Q.submit([&](sycl::handler &cgh) {
    nd_range<1> ndr{1, 1};
    syclexp::work_group_memory<DataT> mem{syclexp::indeterminate};
    cgh.parallel_for(ndr, [=](nd_item<1> it) {
      const auto mem1 = mem;
      // since mem1 is const, all of the following should succeed.
      if constexpr (!std::is_array_v<DataT>)
        mem1 = DataT{};
      convertToDataT(mem1);
      const auto *ptr = &mem1;
      const auto &mptr = mem1.template get_multi_ptr<>();
    });
  });
}

template <typename DataT>
void test_helper(syclexp::work_group_memory<DataT> mem) {
  static_assert(
      std::is_same_v<typename syclexp::work_group_memory<DataT>::value_type,
                     std::remove_all_extents_t<DataT>>);
  syclexp::work_group_memory<DataT> dummy{mem};
  mem = dummy;
  Q.submit([&](sycl::handler &cgh) {
    if constexpr (sycl::detail::is_unbounded_array_v<DataT>)
      mem = syclexp::work_group_memory<DataT>{1, cgh};
    else
      mem = syclexp::work_group_memory<DataT>{cgh};
    nd_range<1> ndr{1, 1};
    cgh.parallel_for(ndr, [=](nd_item<1> it) {
      convertToDataT(mem);
      if constexpr (!std::is_array_v<DataT>)
        mem = DataT{};
      static_assert(
          std::is_same_v<
              multi_ptr<typename syclexp::work_group_memory<DataT>::value_type,
                        access::address_space::local_space,
                        access::decorated::no>,
              decltype(mem.template get_multi_ptr<access::decorated::no>())>);
      static_assert(
          std::is_same_v<
              multi_ptr<typename syclexp::work_group_memory<DataT>::value_type,
                        access::address_space::local_space,
                        access::decorated::no>,
              decltype(mem.template get_multi_ptr<>())>);
      static_assert(
          std::is_same_v<
              multi_ptr<typename syclexp::work_group_memory<DataT>::value_type,
                        access::address_space::local_space,
                        access::decorated::no>,
              decltype(mem.template get_multi_ptr<access::decorated::no>())>);

      static_assert(
          std::is_same_v<
              multi_ptr<typename syclexp::work_group_memory<DataT>::value_type,
                        access::address_space::local_space,
                        access::decorated::yes>,
              decltype(mem.template get_multi_ptr<access::decorated::yes>())>);
    });
  });
}

template <typename Type, typename... rest> void test() {
  syclexp::work_group_memory<Type> mem{syclexp::indeterminate};
  test_constness<Type>();
  test_helper(mem);
  if constexpr (sizeof...(rest))
    test<rest...>();
}

int main() {
  test<char, int16_t, int, double, half>();
  test<marray<int, 1>, marray<int, 2>, marray<int, 8>>();
  test<vec<int, 1>, vec<int, 2>, vec<int, 8>>();
  test<char *, int16_t *, int *, double *, half *>();
  test<S, U>();
  test<char[1], char[2], int[1], int[2]>();
  test<double[], half[]>();
  return 0;
}
