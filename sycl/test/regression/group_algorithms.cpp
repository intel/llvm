// RUN: %clangxx -fsycl -fsyntax-only %s
// RUN: %if preview-breaking-changes-supported %{ %clangxx -fsycl -fsyntax-only -fpreview-breaking-changes %s %}

#include <sycl/sycl.hpp>

#include <type_traits>
#include <utility>

namespace syclex = sycl::ext::oneapi::experimental;

template <typename T> struct GetElemT {
  using type = T;
};
template <typename T, int N> struct GetElemT<sycl::vec<T, N>> {
  using type = T;
};
template <typename T, size_t N> struct GetElemT<sycl::marray<T, N>> {
  using type = T;
};

template <typename T> struct IsVec : std::false_type {};
template <typename T, int N> struct IsVec<sycl::vec<T, N>> : std::true_type {};

template <typename T> struct IsMArray : std::false_type {};
template <typename T, size_t N>
struct IsMArray<sycl::marray<T, N>> : std::true_type {};

template <typename T, typename OpT, typename GroupT>
void TestReduceAndScan(GroupT Group) {
  T Val{};
  OpT Op{};
  decltype(Op(Val, Val)) Out{};

  std::ignore = sycl::joint_reduce(Group, &Val, &Val, Val, Op);
  std::ignore = sycl::reduce_over_group(Group, Val, Val, Op);

  std::ignore = sycl::joint_exclusive_scan(Group, &Val, &Val, &Out, Val, Op);
  std::ignore = sycl::exclusive_scan_over_group(Group, Val, Val, Op);

  std::ignore = sycl::joint_inclusive_scan(Group, &Val, &Val, &Out, Op, Val);
  std::ignore = sycl::inclusive_scan_over_group(Group, Val, Op, Val);

  if constexpr (sycl::has_known_identity_v<OpT, T>) {
    std::ignore = sycl::joint_reduce(Group, &Val, &Val, Op);
    std::ignore = sycl::reduce_over_group(Group, Val, Op);

    std::ignore = sycl::joint_exclusive_scan(Group, &Val, &Val, &Out, Op);
    std::ignore = sycl::exclusive_scan_over_group(Group, Val, Op);

    std::ignore = sycl::joint_inclusive_scan(Group, &Val, &Val, &Out, Op);
    std::ignore = sycl::inclusive_scan_over_group(Group, Val, Op);
  }
}

template <typename T, typename GroupT> void Test(GroupT Group) {
  T Val{};

  std::ignore = sycl::group_broadcast(Group, Val);
  std::ignore =
      sycl::group_broadcast(Group, Val, typename GroupT::linear_id_type{0});
  std::ignore = sycl::group_broadcast(Group, Val, typename GroupT::id_type{0});

  std::ignore =
      sycl::joint_any_of(Group, &Val, &Val, [](const T &) { return true; });
  std::ignore = sycl::any_of_group(Group, Val, [](const T &) { return true; });
  std::ignore = sycl::any_of_group(Group, [](const T &) { return true; });

  std::ignore =
      sycl::joint_all_of(Group, &Val, &Val, [](const T &) { return true; });
  std::ignore = sycl::all_of_group(Group, Val, [](const T &) { return true; });
  std::ignore = sycl::all_of_group(Group, [](const T &) { return true; });

  std::ignore =
      sycl::joint_none_of(Group, &Val, &Val, [](const T &) { return true; });
  std::ignore = sycl::none_of_group(Group, Val, [](const T &) { return true; });
  std::ignore = sycl::none_of_group(Group, [](const T &) { return true; });

  if constexpr (std::is_same_v<GroupT, sycl::sub_group> ||
                syclex::is_user_constructed_group_v<GroupT>) {
    std::ignore = sycl::shift_group_left(Group, Val);
    std::ignore = sycl::shift_group_right(Group, Val);

    std::ignore = sycl::permute_group_by_xor(Group, Val, 0);

    std::ignore = sycl::select_from_group(Group, Val, 0);
  }

  // It is unclear from the specification whether vec and marray are allowed in
  // reduce and scan. Until that is cleared up we exclude these.
  // See https://github.com/KhronosGroup/SYCL-Docs/issues/461.
  if constexpr (std::is_same_v<typename GetElemT<T>::type, T>) {
    using ElemT = typename GetElemT<T>::type;

    // sycl::logical_and and sycl::logical_or requires the input type to be the
    // same as what && and || returns respectively, so we limit the types.
    if constexpr (std::is_same_v<decltype(Val && Val), T>)
      TestReduceAndScan<T, sycl::logical_and<T>>(Group);
    if constexpr (std::is_same_v<decltype(Val || Val), T>)
      TestReduceAndScan<T, sycl::logical_or<T>>(Group);

    if constexpr (!std::is_same_v<ElemT, bool>) {
      TestReduceAndScan<T, sycl::plus<T>>(Group);
      TestReduceAndScan<T, sycl::multiplies<T>>(Group);
      TestReduceAndScan<T, sycl::minimum<T>>(Group);
      TestReduceAndScan<T, sycl::maximum<T>>(Group);
      if constexpr (std::is_integral_v<ElemT>) {
        TestReduceAndScan<T, sycl::bit_and<T>>(Group);
        TestReduceAndScan<T, sycl::bit_or<T>>(Group);
        TestReduceAndScan<T, sycl::bit_xor<T>>(Group);
      }
    }
  }
}

template <typename T, typename GroupT> void TestForScalarType(GroupT Group) {
  Test<T>(Group);
  Test<sycl::vec<T, 1>>(Group);
  Test<sycl::vec<T, 2>>(Group);
  Test<sycl::vec<T, 3>>(Group);
  Test<sycl::vec<T, 4>>(Group);
  Test<sycl::vec<T, 8>>(Group);
  Test<sycl::vec<T, 16>>(Group);
  Test<sycl::marray<T, 3>>(Group);
  Test<sycl::marray<T, 11>>(Group);
  Test<sycl::marray<T, 40>>(Group);
}

template <typename GroupT> void TestForGroup(GroupT Group) {
  // group_barrier does not take a value, so we test it here.
  sycl::group_barrier(Group);

  TestForScalarType<bool>(Group);
  TestForScalarType<char>(Group);
  TestForScalarType<signed char>(Group);
  TestForScalarType<unsigned char>(Group);
  TestForScalarType<short>(Group);
  TestForScalarType<unsigned short>(Group);
  TestForScalarType<int>(Group);
  TestForScalarType<unsigned int>(Group);
  TestForScalarType<long>(Group);
  TestForScalarType<unsigned long>(Group);
  TestForScalarType<long long>(Group);
  TestForScalarType<unsigned long long>(Group);
  TestForScalarType<sycl::half>(Group);
  TestForScalarType<float>(Group);
  TestForScalarType<double>(Group);
}

int main() {
  sycl::queue Q;
  Q.parallel_for(sycl::nd_range<1>{sycl::range<1>{1}, sycl::range<1>{1}},
                 [=](sycl::nd_item<1> NDI) {
                   sycl::sub_group SG = NDI.get_sub_group();
                   TestForGroup(SG);
                   TestForGroup(NDI.get_group());
                   TestForGroup(syclex::get_ballot_group(SG, true));
                   TestForGroup(syclex::get_fixed_size_group<8>(SG));
                   TestForGroup(syclex::get_tangle_group(SG));
                   TestForGroup(syclex::this_kernel::get_opportunistic_group());
                 });
  return 0;
}
