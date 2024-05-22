// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu

// This e2e test checks the alignment of the annotated USM allocation (host &
// device) in various cases

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_device.hpp>
#include <sycl/ext/oneapi/experimental/annotated_usm/alloc_host.hpp>

#include <complex>
#include <numeric>

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

  auto MDevice = [&](auto... args) {
    return malloc_device_annotated(N, args...).get();
  };
  auto MHost = [&](auto... args) {
    return malloc_host_annotated(N, args...).get();
  };

  auto MAnnotated = [&](auto... args) {
    return malloc_annotated(N, args...).get();
  };

  auto ADevice = [&](size_t align, auto... args) {
    return aligned_alloc_device_annotated(align, N, args...).get();
  };
  auto AHost = [&](size_t align, auto... args) {
    return aligned_alloc_host_annotated(align, N, args...).get();
  };

  auto AAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc_annotated(align, N, args...).get();
  };

  CheckAlignAll(
      1,
      std::tuple{
          // Case: `malloc_xxx` with no alignment constraint, this includes
          // no alignment property or `alignment<0>`
          [&]() { return MDevice(q); },
          [&]() { return MDevice(q, ALN0); },
          [&]() { return MDevice(dev, Ctx); },
          [&]() { return MDevice(dev, Ctx, ALN0); },
          [&]() { return MHost(q); },
          [&]() { return MHost(q, ALN0); },
          [&]() { return MHost(Ctx); },
          [&]() { return MHost(Ctx, ALN0); },
          [&]() { return MAnnotated(dev, Ctx, alloc::device); },
          [&]() { return MAnnotated(dev, Ctx, alloc::device, ALN0); },
          [&]() { return MAnnotated(dev, Ctx, alloc::host); },
          [&]() { return MAnnotated(dev, Ctx, alloc::host, ALN0); },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_device}); },
          [&]() {
            return MAnnotated(dev, Ctx,
                              properties{usm_kind_device, alignment<0>});
          },
          [&]() { return MAnnotated(dev, Ctx, properties{usm_kind_host}); },
          [&]() {
            return MAnnotated(dev, Ctx,
                              properties{usm_kind_host, alignment<0>});
          }

          // Case: `aligned_alloc_xxx` with no alignment constraint, this
          // includes
          // 1. no alignment property or `alignment<0>`
          // 2. alignment argument is 0
          ,
          [&]() { return ADevice(0, q); },
          [&]() { return ADevice(0, q, ALN0); },
          [&]() { return ADevice(0, dev, Ctx); },
          [&]() { return ADevice(0, dev, Ctx, ALN0); },
          [&]() { return AHost(0, q); },
          [&]() { return AHost(0, q, ALN0); },
          [&]() { return AHost(0, Ctx); },
          [&]() { return AHost(0, Ctx, ALN0); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::device); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::device, ALN0); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::host); },
          [&]() { return AAnnotated(0, dev, Ctx, alloc::host, ALN0); }});

  // Case: `aligned_alloc_xxx` with alignment argument (power of 2) and no
  // alignment property
  CheckAlignAll(
      align,
      std::tuple{[&]() { return ADevice(align, q); },
                 [&]() { return ADevice(align, dev, Ctx); },
                 [&]() { return AHost(align, q); },
                 [&]() { return AHost(align, Ctx); },
                 [&]() { return AAnnotated(align, dev, Ctx, alloc::device); },
                 [&]() { return AAnnotated(align, dev, Ctx, alloc::host); }});

  // Case: `malloc_xxx<T>` with no alignment constraint, the alignment is
  // sizeof(T)
  auto TDevice = [&](auto... args) {
    return malloc_device_annotated<T>(N, args...).get();
  };
  auto THost = [&](auto... args) {
    return malloc_host_annotated<T>(N, args...).get();
  };
  auto TAnnotated = [&](auto... args) {
    return malloc_annotated<T>(N, args...).get();
  };

  auto ATDevice = [&](size_t align, auto... args) {
    return aligned_alloc_device_annotated<T>(align, N, args...).get();
  };
  auto ATHost = [&](size_t align, auto... args) {
    return aligned_alloc_host_annotated<T>(align, N, args...).get();
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
          [&]() { return TDevice(q); },
          [&]() { return TDevice(q, ALN0); },
          [&]() { return TDevice(dev, Ctx); },
          [&]() { return TDevice(dev, Ctx, ALN0); },
          [&]() { return THost(q); },
          [&]() { return THost(q, ALN0); },
          [&]() { return THost(Ctx); },
          [&]() { return THost(Ctx, ALN0); },
          [&]() { return TAnnotated(dev, Ctx, alloc::device); },
          [&]() { return TAnnotated(dev, Ctx, alloc::device, ALN0); },
          [&]() { return TAnnotated(dev, Ctx, alloc::host); },
          [&]() { return TAnnotated(dev, Ctx, alloc::host, ALN0); },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_device}); },
          [&]() {
            return TAnnotated(dev, Ctx,
                              properties{usm_kind_device, alignment<0>});
          },
          [&]() { return TAnnotated(dev, Ctx, properties{usm_kind_host}); },
          [&]() {
            return TAnnotated(dev, Ctx,
                              properties{usm_kind_host, alignment<0>});
          }

          // Case: `aligned_alloc_xxx<T>` with no alignment constraint, this
          // includes
          // 1. no alignment property or `alignment<0>`
          // 2. alignment argument is 0
          ,
          [&]() { return ATDevice(0, q); },
          [&]() { return ATDevice(0, q, ALN0); },
          [&]() { return ATDevice(0, dev, Ctx); },
          [&]() { return ATDevice(0, dev, Ctx, ALN0); },
          [&]() { return ATHost(0, q); },
          [&]() { return ATHost(0, q, ALN0); },
          [&]() { return ATHost(0, Ctx); },
          [&]() { return ATHost(0, Ctx, ALN0); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::device); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::device, ALN0); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::host); },
          [&]() { return ATAnnotated(0, dev, Ctx, alloc::host, ALN0); },
      });

  // Case: malloc_xxx<T> with compile-time alignment<K> (K is a power of 2), the
  // alignment is the least-common-multiple of K and sizeof(T)
  CheckAlignLcmAll(
      {{2, sizeof(T)},
       {4, sizeof(T)},
       {8, sizeof(T)},
       {16, sizeof(T)},
       {128, sizeof(T)},
       {256, sizeof(T)},
       {16, sizeof(T)},
       {64, sizeof(T)}},
      std::tuple{[&]() { return TDevice(q, ALN2); },
                 [&]() { return TDevice(dev, Ctx, ALN4); },
                 [&]() { return THost(q, ALN8); },
                 [&]() { return THost(Ctx, ALN16); },
                 [&]() { return TAnnotated(dev, Ctx, alloc::device, ALN128); },
                 [&]() { return TAnnotated(dev, Ctx, alloc::host, ALN256); },
                 [&]() {
                   return TAnnotated(
                       dev, Ctx, properties{usm_kind_device, alignment<16>});
                 },
                 [&]() {
                   return TAnnotated(dev, Ctx,
                                     properties{usm_kind_host, alignment<64>});
                 }});

  // Case: aligned_alloc_xxx<T> with alignment argument K (K is a power of 2)
  // and no alignment properties, the alignment is the least-common-multiple of
  // K and sizeof(T)
  CheckAlignAll(
      std::lcm(align, sizeof(T)),
      std::tuple{[&]() { return ATDevice(align, q); },
                 [&]() { return ATDevice(align, dev, Ctx); },
                 [&]() { return ATHost(align, q); },
                 [&]() { return ATHost(align, Ctx); },
                 [&]() { return ATAnnotated(align, dev, Ctx, alloc::device); },
                 [&]() { return ATAnnotated(align, dev, Ctx, alloc::host); }});

  // Case: aligned_alloc_xxx<T> with alignment argument K (K is a power of 2)
  // and no alignment properties, the alignment is the least-common-multiple of
  // K and sizeof(T)
  CheckAlignLcmAll(
      {
          {2, sizeof(T)},
          {4, sizeof(T)},
          {8, sizeof(T)},
          {16, sizeof(T)},
          {128, sizeof(T)},
          {256, sizeof(T)},
      },
      std::tuple{
          [&]() { return ATDevice(2, q); },
          [&]() { return ATDevice(4, dev, Ctx, ALN4); },
          [&]() { return ATHost(8, q, ALN8); },
          [&]() { return ATHost(16, Ctx, ALN16); },
          [&]() { return ATAnnotated(128, dev, Ctx, alloc::device, ALN128); },
          [&]() { return ATAnnotated(256, dev, Ctx, alloc::host, ALN256); }});

  // Case: aligned_alloc_xxx<T> with compile-time alignment<K> (K is a power of
  // 2), the alignment is the least-common-multiple of the alignment argument (a
  // power of 2), K and sizeof(T)
  CheckAlignLcmAll(
      {
          {align, 2, sizeof(T)},
          {align, 4, sizeof(T)},
          {align, 8, sizeof(T)},
          {align, 16, sizeof(T)},
          {align, 128, sizeof(T)},
          {align, 256, sizeof(T)},
      },
      std::tuple{
          [&]() { return ATDevice(align, q, ALN2); },
          [&]() { return ATDevice(align, dev, Ctx, ALN4); },
          [&]() { return ATHost(align, q, ALN8); },
          [&]() { return ATHost(align, Ctx, ALN16); },
          [&]() { return ATAnnotated(align, dev, Ctx, alloc::device, ALN128); },
          [&]() { return ATAnnotated(align, dev, Ctx, alloc::host, ALN256); }});

  // Test cases that are expected to return null
  auto check_null = [&q](auto AllocFn, int Line = __builtin_LINE(),
                         int Case = 0) {
    decltype(AllocFn()) Ptr = AllocFn();
    auto v = reinterpret_cast<uintptr_t>(Ptr);
    if (v != 0) {
      free(Ptr, q);
      std::cout << "Failed at line " << Line << ", case " << Case << std::endl;
      assert(false && "The return is not null!");
    }
  };

  auto CheckNullAll = [&](auto Funcs, int Line = __builtin_LINE()) {
    std::apply(
        [&](auto... Fs) {
          int Case = 0;
          (void)std::initializer_list<int>{
              (check_null(Fs, Line, Case++), 0)...};
        },
        Funcs);
  };

  CheckNullAll(std::tuple{
      // Case: malloc_xxx with compile-time alignment<K> (K is not a power of
      // 2),
      // nullptr is returned
      [&]() { return MDevice(q, properties{alignment<5>}); },
      [&]() { return MDevice(dev, Ctx, properties{alignment<10>}); },
      [&]() { return MHost(q, properties{alignment<25>}); },
      [&]() { return MHost(Ctx, properties{alignment<50>}); },
      [&]() {
        return MAnnotated(q, alloc::device, properties{alignment<127>});
      },
      [&]() {
        return MAnnotated(dev, Ctx, alloc::host, properties{alignment<200>});
      },
      [&]() {
        return MAnnotated(q, properties{usm_kind_device, alignment<500>});
      },
      [&]() {
        return MAnnotated(dev, Ctx, properties{usm_kind_host, alignment<1000>});
      }

      // Case: malloc_xxx<T> with compile-time alignment<K> (K is not a power of
      // 2),
      // nullptr is returned
      ,
      [&]() { return TDevice(q, properties{alignment<75>}); },
      [&]() { return TDevice(dev, Ctx, properties{alignment<100>}); },
      [&]() { return THost(q, properties{alignment<205>}); },
      [&]() { return THost(Ctx, properties{alignment<500>}); },
      [&]() {
        return TAnnotated(q, alloc::device, properties{alignment<127>});
      },
      [&]() {
        return TAnnotated(dev, Ctx, alloc::host, properties{alignment<200>});
      },
      [&]() {
        return TAnnotated(q, properties{usm_kind_device, alignment<500>});
      },
      [&]() {
        return TAnnotated(dev, Ctx, properties{usm_kind_host, alignment<1000>});
      }
      // Case: aligned_alloc_xxx with no alignment property, and the alignment
      // argument is not a power of 2, the result is nullptr
      ,
      [&]() { return ADevice(3, q); },
      [&]() { return ADevice(7, dev, Ctx); },
      [&]() { return AHost(7, q); },
      [&]() { return AHost(9, Ctx); },
      [&]() { return AAnnotated(15, q, alloc::device); },
      [&]() { return AAnnotated(17, dev, Ctx, alloc::host); }

      // Case: aligned_alloc_xxx<T> with no alignment property, and the
      // alignment
      // argument is not a power of 2, the result is nullptr
      ,
      [&]() { return ATDevice(65, q); },
      [&]() { return ATDevice(70, dev, Ctx); },
      [&]() { return ATHost(174, q); },
      [&]() { return ATHost(299, Ctx); },
      [&]() { return ATAnnotated(550, q, alloc::device); },
      [&]() { return ATAnnotated(1700, dev, Ctx, alloc::host); }});
}

struct alignas(64) MyStruct {
  int x;
};

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  testAlign<int>(q, 128);
  testAlign<std::complex<double>>(q, 4);
  testAlign<MyStruct>(q, 4);
  return 0;
}
