// REQUIRES: aspect-usm_shared_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu

// This e2e test checks the alignment of the annotated shared USM allocation in
// various cases

#include <sycl/sycl.hpp>

#include <complex>
#include <numeric>

// clang-format on

using namespace sycl::ext::oneapi::experimental;
using namespace sycl::ext::intel::experimental;
using alloc = sycl::usm::alloc;

template <typename T> void testAlign(sycl::queue &q, unsigned align) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  properties ALN0{alignment<0>};
  properties ALN2{alignment<2>};
  properties ALN4{alignment<4>};
  properties ALN8{alignment<8>};
  properties ALN16{alignment<16>};
  properties ALN32{alignment<32>};
  properties ALN64{alignment<64>};
  properties ALN128{alignment<128>};
  properties ALN256{alignment<256>};
  properties ALN512{alignment<512>};

  auto check_align = [&q](size_t Alignment, auto AllocFn,
                          int Line = __builtin_LINE(), int Case = 0) {
    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Arr[10];
    for (auto *&Elem : Arr)
      Elem = AllocFn();
    for (auto *Ptr : Arr) {
      auto v = reinterpret_cast<uintptr_t>(Ptr);
      if ((v & (Alignment - 1)) != 0) {
        std::cout << "Failed at line " << Line << ", case " << Case
                  << std::endl;
        assert(false && "Not properly aligned!");
        break; // To be used with commented out assert above.
      }
    }
    for (auto *Ptr : Arr)
      free(Ptr, q);
  };

  auto CheckAlignAll = [&](size_t Expected, auto Funcs,
                           int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check_align(Expected, Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  constexpr int N = 10;
  assert(align > 0 || (align & (align - 1)) == 0);

  auto MShared = [&](auto... args) {
    return malloc_shared_annotated(N, args...).get();
  };
  auto MAnnotated = [&](auto... args) {
    return malloc_annotated(N, args...).get();
  };

  auto AShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared_annotated(align, N, args...).get();
  };
  auto AAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated(align, N, args...).get();
  };

  CheckAlignAll(
      1,
      std::tuple{
          // Case: `malloc_xxx` with no alignment constraint, this includes
          // no alignment property or `alignment<0>`
          [&]() { return MShared(q); }, [&]() { return MShared(q, ALN0); },
          [&]() { return MShared(dev, Ctx); },
          [&]() { return MShared(dev, Ctx, ALN0); },
          [&]() { return MAnnotated(dev, Ctx, alloc::shared); },
          [&]() { return MAnnotated(dev, Ctx, alloc::shared, ALN0); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_shared}); },
          [&]() {
            return MAnnotated(dev, Ctx,
                              properties{usm_kind_shared, alignment<0>});
          }

          // Case: `aligned_alloc_xxx` with no alignment constraint, this
          // includes
          // 1. no alignment property or `alignment<0>`
          // 2. alignment argument is 0
          ,
          [&]() { return AShared(0, q); },
          [&]() { return AShared(0, q, ALN0); },
          [&]() { return AShared(0, dev, Ctx); },
          [&]() { return AShared(0, dev, Ctx, ALN0); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::shared); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::shared, ALN0); }});

  // Case: `aligned_alloc_xxx` with alignment argument (power of 2) and no
  // alignment property
  CheckAlignAll(
      align,
      std::tuple{[&]() { return AShared(align, q); },
                 [&]() { return AShared(align, dev, Ctx); },
                 [&]() { return AAnnotated(align, dev, Ctx, alloc::shared); }});

  // Case: `malloc_xxx<T>` with no alignment constraint, the alignment is
  // sizeof(T)
  auto TShared = [&](auto... args) {
    return malloc_shared_annotated<T>(N, args...).get();
  };
  auto TAnnotated = [&](auto... args) {
    return malloc_annotated<T>(N, args...).get();
  };

  auto ATShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared_annotated<T>(align, N, args...).get();
  };
  auto ATAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated<T>(align, N, args...).get();
  };

  auto check_align_lcm = [&q](std::vector<size_t> Alignments, auto AllocFn,
                              int Line = __builtin_LINE(), int Case = 0) {
    assert(Alignments.size() > 0);
    size_t Lcm = Alignments[0];
    for (int i = 1; i < Alignments.size(); i++)
      Lcm = std::lcm(Lcm, Alignments[i]);

    // First allocation might naturally be over-aligned. Do several of them to
    // do the verification;
    decltype(AllocFn()) Arr[10];
    for (auto *&Elem : Arr)
      Elem = AllocFn();
    for (auto *Ptr : Arr) {
      auto v = reinterpret_cast<uintptr_t>(Ptr);
      if ((v & (Lcm - 1)) != 0) {
        std::cout << "Failed at line " << Line << ", case " << Case
                  << std::endl;
        assert(false && "Not properly aligned!");
        break; // To be used with commented out assert above.
      }
    }
    for (auto *Ptr : Arr)
      free(Ptr, q);
  };

  auto CheckAlignLcmAll = [&](std::vector<std::vector<size_t>> AlignVec,
                              auto Funcs, int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0, Idx = 0;
          (void)std::initializer_list<int>{
              (check_align_lcm(AlignVec[Idx++], Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  CheckAlignAll(
      sizeof(T),
      std::tuple{
          // Case: `malloc_xxx<T>` with no alignment constraint, this includes
          // no alignment property or `alignment<0>`
          [&]() { return TShared(q); }, [&]() { return TShared(q, ALN0); },
          [&]() { return TShared(dev, Ctx); },
          [&]() { return TShared(dev, Ctx, ALN0); },
          [&]() { return TAnnotated(dev, Ctx, alloc::shared); },
          [&]() { return TAnnotated(dev, Ctx, alloc::shared, ALN0); },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_shared}); },
          [&]() {
            return TAnnotated(dev, Ctx,
                              properties{usm_kind_shared, alignment<0>});
          }

          // Case: `aligned_alloc_xxx<T>` with no alignment constraint, this
          // includes
          // 1. no alignment property or `alignment<0>`
          // 2. alignment argument is 0
          ,
          [&]() { return ATShared(0, q); },
          [&]() { return ATShared(0, q, ALN0); },
          [&]() { return ATShared(0, dev, Ctx); },
          [&]() { return ATShared(0, dev, Ctx, ALN0); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::shared); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::shared, ALN0); }});

  // Case: malloc_xxx<T> with compile-time alignment<K> (K is a power of 2), the
  // alignment is the least-common-multiple of K and sizeof(T)
  CheckAlignLcmAll(
      {{32, sizeof(T)}, {64, sizeof(T)}, {512, sizeof(T)}, {256, sizeof(T)}},
      std::tuple{[&]() { return TShared(q, ALN32); },
                 [&]() { return TShared(dev, Ctx, ALN64); },
                 [&]() { return TAnnotated(dev, Ctx, alloc::shared, ALN512); },
                 [&]() {
                   return TAnnotated(
                       dev, Ctx, properties{usm_kind_shared, alignment<256>});
                 }});

  // Case: aligned_alloc_xxx<T> with alignment argument K (K is a power of 2)
  // and no alignment properties, the alignment is the least-common-multiple of
  // K and sizeof(T)
  CheckAlignLcmAll(
      {
          {32, sizeof(T)},
          {64, sizeof(T)},
          {512, sizeof(T)},
      },
      std::tuple{
          [&]() { return ATShared(32, q, ALN32); },
          [&]() { return ATShared(64, dev, Ctx, ALN64); },
          [&]() { return ATAnnotated(512, dev, Ctx, alloc::shared, ALN512); }});

  // Case: aligned_alloc_xxx<T> with alignment argument K (K is a power of 2)
  // and no alignment properties, the alignment is the least-common-multiple of
  // K and sizeof(T)
  CheckAlignAll(std::lcm(align, sizeof(T)),
                std::tuple{[&]() { return ATShared(align, q); },
                           [&]() { return ATShared(align, dev, Ctx); },
                           [&]() {
                             return ATAnnotated(align, dev, Ctx, alloc::shared);
                           }});

  // Case: aligned_alloc_xxx<T> with compile-time alignment<K> (K is a power of
  // 2), the alignment is the least-common-multiple of the alignment argument (a
  // power of 2), K and sizeof(T)
  CheckAlignLcmAll(
      {{align, 32, sizeof(T)}, {align, 64, sizeof(T)}, {align, 512, sizeof(T)}},
      std::tuple{[&]() { return ATShared(align, q, ALN32); },
                 [&]() { return ATShared(align, dev, Ctx, ALN64); },
                 [&]() {
                   return ATAnnotated(align, dev, Ctx, alloc::shared, ALN512);
                 }});
}

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  testAlign<int>(q, 128);
  testAlign<std::complex<double>>(q, 4);
  return 0;
}