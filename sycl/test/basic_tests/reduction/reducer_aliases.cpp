// RUN: %clangxx -fsycl -fsyntax-only -sycl-std=2020 %s

// Tests the member aliases on the reducer class.

#include <sycl/sycl.hpp>

#include <type_traits>

template <typename T> class Kernel;

template <typename ReducerT, typename ValT, typename BinOp, int Dims>
void CheckReducerAliases() {
  static_assert(std::is_same_v<typename ReducerT::value_type, ValT>);
  static_assert(std::is_same_v<typename ReducerT::binary_operation, BinOp>);
  static_assert(ReducerT::dimensions == Dims);
}

template <typename T> void CheckAllReducers(sycl::queue &Q) {
  T *Vals = sycl::malloc_device<T>(4, Q);
  sycl::span<T, 4> SpanVal(Vals, 4);

  auto CustomOp = [](const T &LHS, const T &RHS) { return LHS + RHS; };

  auto ValReduction1 = sycl::reduction(Vals, sycl::plus<>());
  auto ValReduction2 = sycl::reduction(Vals, T{}, sycl::plus<>());
  auto ValReduction3 = sycl::reduction(Vals, T{}, CustomOp);
  auto SpanReduction1 = sycl::reduction(SpanVal, sycl::plus<>());
  auto SpanReduction2 = sycl::reduction(SpanVal, T{}, sycl::plus<>());
  auto SpanReduction3 = sycl::reduction(SpanVal, T{}, CustomOp);
  // TODO: Add cases with identityless reductions when supported.
  Q.parallel_for<Kernel<T>>(
      sycl::range<1>{10}, ValReduction1, ValReduction2, ValReduction3,
      SpanReduction1, SpanReduction2, SpanReduction3,
      [=](sycl::id<1>, auto &ValRedu1, auto &ValRedu2, auto &ValRedu3,
          auto &SpanRedu1, auto &SpanRedu2, auto &SpanRedu3) {
        CheckReducerAliases<std::remove_reference_t<decltype(ValRedu1)>, T,
                            sycl::plus<>, 0>();
        CheckReducerAliases<std::remove_reference_t<decltype(ValRedu2)>, T,
                            sycl::plus<>, 0>();
        CheckReducerAliases<std::remove_reference_t<decltype(ValRedu3)>, T,
                            decltype(CustomOp), 0>();
        CheckReducerAliases<std::remove_reference_t<decltype(SpanRedu1)>, T,
                            sycl::plus<>, 1>();
        CheckReducerAliases<std::remove_reference_t<decltype(SpanRedu2)>, T,
                            sycl::plus<>, 1>();
        CheckReducerAliases<std::remove_reference_t<decltype(SpanRedu3)>, T,
                            decltype(CustomOp), 1>();
      });
}

int main() {
  sycl::queue Q;
  CheckAllReducers<char>(Q);
  CheckAllReducers<short>(Q);
  CheckAllReducers<int>(Q);
  CheckAllReducers<long>(Q);
  CheckAllReducers<float>(Q);
  CheckAllReducers<double>(Q);
  CheckAllReducers<sycl::half>(Q);
  return 0;
}
