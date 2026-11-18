// RUN: %clangxx -fsycl -fsyntax-only -ferror-limit=0 -Xclang -verify %s -Xclang -verify-ignore-unexpected=note

// Tests the errors emitted from using the deleted copy and move assignment
// operators and constructors.

// NOTE: This is a negative test that asserts the semantic invariant that a
// 'sycl::reducer' can be neither copied nor moved. The primary expected
// diagnostic is 'call to deleted constructor'. The exact number of diagnostics
// reported from sycl/reduction.hpp is not part of this test's contract, since
// it depends on how the selected strategy lowers the reduction:
//
// - A reduction strategy may compile several kernel variants (e.g. single- vs.
//   multi-work-group, or different predicate accessor types). Each variant
//   calls the user lambda and reports the error, so the deleted-constructor
//   message can appear more than once for the same call.
// - For range reductions the user lambda is called from a generic wrapper
//   lambda, so the by-value copy fails inside the wrapper's body. That
//   invalidates the wrapper's call operator, and each further kernel variant
//   that calls it also reports 'no matching function for call to object of
//   type'. The cascade is the natural result of two kernel variants sharing one
//   wrapper. The number of these cascades depends on how many functor variants
//   are instantiated.
//
// The checks require the semantic failure to be seen and tolerate the
// follow-on diagnostics, avoiding brittleness across refactors.

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

  // Follow-on of 'call to deleted constructor' for range reductions; the number
  // of occurrences depends on the number of kernel variants, see NOTE above.
  // expected-error-re@sycl/reduction.hpp:* 0+ {{no matching function for call to object of type 'const (lambda at {{.*}})'}}

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
