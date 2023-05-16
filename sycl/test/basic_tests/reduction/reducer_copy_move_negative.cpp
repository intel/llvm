// RUN: %clangxx -fsycl -fsyntax-only -ferror-limit=0 -Xclang -verify %s -Xclang -verify-ignore-unexpected=note

// Tests the errors emitted from using the deleted copy and move assignment
// operators and constructors.

#include <sycl/sycl.hpp>

#include <type_traits>

template <class T> struct PlusWithoutIdentity {
  T operator()(const T &A, const T &B) const { return A + B; }
};

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

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, std::plus<int>, 0{{.*}}>'}}
  Q.parallel_for(sycl::range<1>{1024}, ScalarRed1,
                 [=](sycl::item<1>, auto Reducer) {});

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, std::plus<int>, 0{{.*}}>'}}
  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, ScalarRed1,
                 [=](sycl::nd_item<1>, auto Reducer) {});

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>'}}
  Q.parallel_for(sycl::range<1>{1024}, ScalarRed2,
                 [=](sycl::item<1>, auto Reducer) {});

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>'}}
  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, ScalarRed2,
                 [=](sycl::nd_item<1>, auto Reducer) {});

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, std::plus<int>, 1{{.*}}>'}}
  Q.parallel_for(sycl::range<1>{1024}, SpanRed1,
                 [=](sycl::item<1>, auto Reducer) {});

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, std::plus<int>, 1{{.*}}>'}}
  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, SpanRed1,
                 [=](sycl::nd_item<1>, auto Reducer) {});

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>'}}
  Q.parallel_for(sycl::range<1>{1024}, SpanRed2,
                 [=](sycl::item<1>, auto Reducer) {});

  // expected-error-re@sycl/reduction.hpp:* {{call to deleted constructor of 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>'}}
  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, SpanRed2,
                 [=](sycl::nd_item<1>, auto Reducer) {});

  Q.parallel_for(sycl::range<1>{1024}, ScalarRed1,
                 [=](sycl::item<1>, auto &Reducer) {
                   using reducer_t = std::remove_reference_t<decltype(Reducer)>;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyAssign = Reducer;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveAssign = std::move(Reducer);
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyCtor{Reducer};
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveCtor{std::move(Reducer)};
                 });

  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, ScalarRed1,
                 [=](sycl::nd_item<1>, auto &Reducer) {
                   using reducer_t = std::remove_reference_t<decltype(Reducer)>;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyAssign = Reducer;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveAssign = std::move(Reducer);
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyCtor{Reducer};
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveCtor{std::move(Reducer)};
                 });

  Q.parallel_for(sycl::range<1>{1024}, ScalarRed2,
                 [=](sycl::item<1>, auto &Reducer) {
                   using reducer_t = std::remove_reference_t<decltype(Reducer)>;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyAssign = Reducer;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveAssign = std::move(Reducer);
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyCtor{Reducer};
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveCtor{std::move(Reducer)};
                 });

  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, ScalarRed2,
                 [=](sycl::nd_item<1>, auto &Reducer) {
                   using reducer_t = std::remove_reference_t<decltype(Reducer)>;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyAssign = Reducer;
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveAssign = std::move(Reducer);
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerCopyCtor{Reducer};
                   // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
                   reducer_t ReducerMoveCtor{std::move(Reducer)};
                 });

  Q.parallel_for(
      sycl::range<1>{1024}, SpanRed1, [=](sycl::item<1>, auto &Reducer) {
        using reducer_t = std::remove_reference_t<decltype(Reducer)>;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyAssign = Reducer;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveAssign = std::move(Reducer);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyCtor{Reducer};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveCtor{std::move(Reducer)};

        using reducer_subscript_t =
            std::remove_reference_t<decltype(Reducer[0])>;
        reducer_subscript_t ReducerSubscript = Reducer[0];
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyAssign = ReducerSubscript;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveAssign =
            std::move(ReducerSubscript);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyCtor{ReducerSubscript};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveCtor{
            std::move(ReducerSubscript)};
      });

  Q.parallel_for(
      sycl::nd_range<1>{1024, 1024}, SpanRed1,
      [=](sycl::nd_item<1>, auto &Reducer) {
        using reducer_t = std::remove_reference_t<decltype(Reducer)>;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyAssign = Reducer;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveAssign = std::move(Reducer);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyCtor{Reducer};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, std::plus<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveCtor{std::move(Reducer)};

        using reducer_subscript_t =
            std::remove_reference_t<decltype(Reducer[0])>;
        reducer_subscript_t ReducerSubscript = Reducer[0];
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyAssign = ReducerSubscript;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveAssign =
            std::move(ReducerSubscript);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyCtor{ReducerSubscript};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, std::plus<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveCtor{
            std::move(ReducerSubscript)};
      });

  Q.parallel_for(
      sycl::range<1>{1024}, SpanRed2, [=](sycl::item<1>, auto &Reducer) {
        using reducer_t = std::remove_reference_t<decltype(Reducer)>;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyAssign = Reducer;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveAssign = std::move(Reducer);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyCtor{Reducer};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveCtor{std::move(Reducer)};

        using reducer_subscript_t =
            std::remove_reference_t<decltype(Reducer[0])>;
        reducer_subscript_t ReducerSubscript = Reducer[0];
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyAssign = ReducerSubscript;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveAssign =
            std::move(ReducerSubscript);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyCtor{ReducerSubscript};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveCtor{
            std::move(ReducerSubscript)};
      });

  Q.parallel_for(
      sycl::nd_range<1>{1024, 1024}, SpanRed2,
      [=](sycl::nd_item<1>, auto &Reducer) {
        using reducer_t = std::remove_reference_t<decltype(Reducer)>;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyAssign = Reducer;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveAssign = std::move(Reducer);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerCopyCtor{Reducer};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 1{{.*}}>')}}
        reducer_t ReducerMoveCtor{std::move(Reducer)};

        using reducer_subscript_t =
            std::remove_reference_t<decltype(Reducer[0])>;
        reducer_subscript_t ReducerSubscript = Reducer[0];
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyAssign = ReducerSubscript;
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveAssign =
            std::move(ReducerSubscript);
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptCopyCtor{ReducerSubscript};
        // expected-error-re@+1 {{call to deleted constructor of 'reducer_subscript_t' (aka 'sycl::reducer<int, PlusWithoutIdentity<int>, 0{{.*}}>')}}
        reducer_subscript_t ReducerSubscriptMoveCtor{
            std::move(ReducerSubscript)};
      });

  return 0;
}
