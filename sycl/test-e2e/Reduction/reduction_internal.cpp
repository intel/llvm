// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/reduction.hpp>

using namespace sycl;

template <int Dims> auto get_global_range(range<Dims> Range) { return Range; }
template <int Dims> auto get_global_range(nd_range<Dims> NDRange) {
  return NDRange.get_global_range();
}

template <int Dims, bool WithOffset>
auto get_global_id(item<Dims, WithOffset> Item) {
  return Item.get_id();
}
template <int Dims> auto get_global_id(nd_item<Dims> NDItem) {
  return NDItem.get_global_id();
}

template <int Dims> auto get_global_id(id<Dims> Id) { return Id; }

// We can select strategy explicitly so no need to test all combinations of
// types/operations.
using T = int;
using BinOpTy = std::plus<T>;

// On Windows, allocating new memory and then initializing it is slow for some
// reason (not related to reductions). Try to re-use the same memory between
// test cases.
struct RedStorage {
  RedStorage(queue &q) : q(q), Ptr(malloc_device<T>(1, q)), Buf(1) {}
  ~RedStorage() { free(Ptr, q); }

  template <bool UseUSM> auto get() {
    if constexpr (UseUSM)
      return Ptr;
    else
      return Buf;
  }
  queue &q;
  T *Ptr;
  buffer<T, 1> Buf;
};

template <bool UseUSM, bool InitToIdentity,
          detail::reduction::strategy Strategy, typename RangeTy>
static void test(RedStorage &Storage, RangeTy Range) {
  queue &q = Storage.q;

  T Init{19};

  auto Red = Storage.get<UseUSM>();
  auto GetRedAcc = [&](handler &cgh) {
    if constexpr (UseUSM)
      return Red;
    else
      return accessor{Red, cgh};
  };

  q.submit([&](handler &cgh) {
     auto RedAcc = GetRedAcc(cgh);
     cgh.single_task([=]() { RedAcc[0] = Init; });
   }).wait();

  q.submit([&](handler &cgh) {
     auto RedSycl = [&]() {
       if constexpr (UseUSM)
         if constexpr (InitToIdentity)
           return reduction(Red, BinOpTy{},
                            property::reduction::initialize_to_identity{});
         else
           return reduction(Red, BinOpTy{});
       else if constexpr (InitToIdentity)
         return reduction(Red, cgh, BinOpTy{},
                          property::reduction::initialize_to_identity{});
       else
         return reduction(Red, cgh, BinOpTy{});
     }();
     detail::reduction_parallel_for<detail::auto_name, Strategy>(
         cgh, Range, ext::oneapi::experimental::empty_properties_t{}, RedSycl,
         [=](auto Item, auto &Red) { Red.combine(T{1}); });
   }).wait();
  sycl::buffer<T> ResultBuf{sycl::range{1}};
  q.submit([&](handler &cgh) {
    sycl::accessor Result{ResultBuf, cgh};
    auto RedAcc = GetRedAcc(cgh);
    cgh.single_task([=]() { Result[0] = RedAcc[0]; });
  });
  sycl::host_accessor Result{ResultBuf};
  auto N = get_global_range(Range).size();
  int Expected = InitToIdentity ? N : Init + N;
#if defined(__PRETTY_FUNCTION__)
  std::cout << __PRETTY_FUNCTION__;
#elif defined(__FUNCSIG__)
  std::cout << __FUNCSIG__;
#endif
  std::cout << ": " << Result[0] << ", expected " << Expected << std::endl;
  assert(Result[0] == Expected);
}

template <int... Inds, class F>
void loop_impl(std::integer_sequence<int, Inds...>, F &&f) {
  (f(std::integral_constant<int, Inds>{}), ...);
}

template <int count, class F> void loop(F &&f) {
  loop_impl(std::make_integer_sequence<int, count>{}, std::forward<F>(f));
}

template <bool UseUSM, bool InitToIdentity, typename RangeTy>
void testAllStrategies(RedStorage &Storage, RangeTy Range) {
  loop<(int)detail::reduction::strategy::multi>([&](auto Id) {
    constexpr auto Strategy =
        // Skip auto_select == 0.
        detail::reduction::strategy{decltype(Id)::value + 1};
    test<UseUSM, InitToIdentity, Strategy>(Storage, Range);
  });
}

int main() {
  queue q;
  RedStorage Storage(q);

  auto TestRange = [&](auto Range) {
    testAllStrategies<true, true>(Storage, Range);
    testAllStrategies<true, false>(Storage, Range);
    testAllStrategies<false, true>(Storage, Range);
    testAllStrategies<false, false>(Storage, Range);
  };

  TestRange(range<1>{42});
  TestRange(range<2>{8, 8});
  TestRange(range<3>{7, 7, 5});
  TestRange(nd_range<1>{range<1>{7}, range<1>{7}});
  TestRange(nd_range<1>{range<1>{3 * 3}, range<1>{3}});

  // TODO: Strategies historically adopted from sycl::range implementation only
  // support 1-Dim case.
  //
  // TestRange(nd_range<2>{range<2>{7, 3}, range<2> {7, 3}});
  // TestRange(nd_range<2>{range<2>{14, 9}, range<2> {7, 3}});
  return 0;
}
