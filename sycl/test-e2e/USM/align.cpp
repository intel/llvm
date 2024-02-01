// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// UNSUPPORTED: gpu

// E2E tests for annotated USM allocation functions with alignment arguments
// that are not powers of 2. Note this test does not work on gpu because some
// tests expect to return nullptr, e.g. when the alignment argument is not a
// power of 2, while the gpu runtime has different behavior

#include <sycl/sycl.hpp>

#include <complex>
#include <numeric>

using namespace sycl;
using namespace ext::oneapi::experimental;
using namespace ext::intel::experimental;
using alloc = usm::alloc;

template <typename T> void testAlign(sycl::queue &q, unsigned align) {
  const sycl::context &Ctx = q.get_context();
  auto dev = q.get_device();

  constexpr int N = 10;
  assert(align > 0 || (align & (align - 1)) == 0);

  auto ADevice = [&](size_t align, auto... args) {
    return aligned_alloc_device(align, N, args...);
  };
  auto AHost = [&](size_t align, auto... args) {
    return aligned_alloc_host(align, N, args...);
  };
  auto AShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared(align, N, args...);
  };
  auto AAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc(align, N, args...);
  };

  auto ATDevice = [&](size_t align, auto... args) {
    return aligned_alloc_device<T>(align, N, args...);
  };
  auto ATHost = [&](size_t align, auto... args) {
    return aligned_alloc_host<T>(align, N, args...);
  };
  auto ATShared = [&](size_t align, auto... args) {
    return aligned_alloc_shared<T>(align, N, args...);
  };
  auto ATAnnotated = [&](size_t align, auto... args) {
    return aligned_alloc<T>(align, N, args...);
  };

  // Test cases that are expected to return null
  auto check_null = [&q](auto AllocFn, int Line, int Case) {
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
      // Case: aligned_alloc_xxx with no alignment property, and the alignment
      // argument is not a power of 2, the result is nullptr
      [&]() { return ADevice(3, q); }, [&]() { return ADevice(5, dev, Ctx); },
      [&]() { return AHost(7, q); }, [&]() { return AHost(9, Ctx); },
      [&]() { return AShared(114, q); },
      [&]() { return AShared(1023, dev, Ctx); },
      [&]() { return AAnnotated(15, q, alloc::device); },
      [&]() { return AAnnotated(17, dev, Ctx, alloc::host); }
      // Case: aligned_alloc_xxx<T> with no alignment property, and the
      // alignment
      // argument is not a power of 2, the result is nullptr
      ,
      [&]() { return ATDevice(3, q); }, [&]() { return ATDevice(5, dev, Ctx); },
      [&]() { return ATHost(7, q); }, [&]() { return ATHost(9, Ctx); },
      [&]() { return ATShared(1919, q); },
      [&]() { return ATShared(11, dev, Ctx); },
      [&]() { return ATAnnotated(15, q, alloc::device); },
      [&]() { return ATAnnotated(17, dev, Ctx, alloc::host); }});
}

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  testAlign<int>(q, 128);
  testAlign<std::complex<double>>(q, 4);
  return 0;
}
