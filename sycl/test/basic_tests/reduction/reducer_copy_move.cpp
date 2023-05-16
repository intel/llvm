// RUN: %clangxx -fsycl -fsyntax-only %s

// Tests that the reducer class is neither movable nor copyable.

#include <sycl/sycl.hpp>

#include <type_traits>

template <class T> struct PlusWithoutIdentity {
  T operator()(const T &A, const T &B) const { return A + B; }
};

template <typename ReducerT> static constexpr void checkReducer() {
  static_assert(!std::is_copy_constructible_v<ReducerT>);
  static_assert(!std::is_move_constructible_v<ReducerT>);
  static_assert(!std::is_copy_assignable_v<ReducerT>);
  static_assert(!std::is_move_assignable_v<ReducerT>);
}

int main() {
  sycl::queue Q;

  int *ScalarMem = sycl::malloc_shared<int>(1, Q);
  int *SpanMem = sycl::malloc_shared<int>(8, Q);
  auto ScalarRed1 = sycl::reduction(ScalarMem, std::plus<int>{});
  auto ScalarRed2 = sycl::reduction(ScalarMem, PlusWithoutIdentity<int>{});
  auto SpanRed1 =
      sycl::reduction(sycl::span<int, 8>{SpanMem, 8}, std::plus<int>{});
  auto SpanRed2 = sycl::reduction(sycl::span<int, 8>{SpanMem, 8},
                                  PlusWithoutIdentity<int>{});

  Q.parallel_for(sycl::range<1>{1024}, ScalarRed1,
                 [=](sycl::item<1>, auto &Reducer) {
                   checkReducer<std::remove_reference_t<decltype(Reducer)>>();
                 });

  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, ScalarRed1,
                 [=](sycl::nd_item<1>, auto &Reducer) {
                   checkReducer<std::remove_reference_t<decltype(Reducer)>>();
                 });

  Q.parallel_for(sycl::range<1>{1024}, ScalarRed2,
                 [=](sycl::item<1>, auto &Reducer) {
                   checkReducer<std::remove_reference_t<decltype(Reducer)>>();
                 });

  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, ScalarRed2,
                 [=](sycl::nd_item<1>, auto &Reducer) {
                   checkReducer<std::remove_reference_t<decltype(Reducer)>>();
                 });

  Q.parallel_for(
      sycl::range<1>{1024}, SpanRed1, [=](sycl::item<1>, auto &Reducer) {
        checkReducer<std::remove_reference_t<decltype(Reducer)>>();
        checkReducer<std::remove_reference_t<decltype(Reducer[0])>>();
      });

  Q.parallel_for(
      sycl::nd_range<1>{1024, 1024}, SpanRed1,
      [=](sycl::nd_item<1>, auto &Reducer) {
        checkReducer<std::remove_reference_t<decltype(Reducer)>>();
        checkReducer<std::remove_reference_t<decltype(Reducer[0])>>();
      });

  Q.parallel_for(
      sycl::range<1>{1024}, SpanRed2, [=](sycl::item<1>, auto &Reducer) {
        checkReducer<std::remove_reference_t<decltype(Reducer)>>();
        checkReducer<std::remove_reference_t<decltype(Reducer[0])>>();
      });

  Q.parallel_for(
      sycl::nd_range<1>{1024, 1024}, SpanRed2,
      [=](sycl::nd_item<1>, auto &Reducer) {
        checkReducer<std::remove_reference_t<decltype(Reducer)>>();
        checkReducer<std::remove_reference_t<decltype(Reducer[0])>>();
      });

  return 0;
}
