// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// E2E tests for aligned USM allocation functions with different alignment
// arguments. Depending on the backend the alignment may or may not be supported
// but either way, according to the SYCL spec, we should see a nullptr if it is
// not supported or we should verify that the pointer returned is indeed
// aligned if it is supported.

#include <sycl/detail/core.hpp>
#include <sycl/usm.hpp>

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
    auto ptr = aligned_alloc_device(align, N, args...);
    assert(!ptr || !(reinterpret_cast<std::uintptr_t>(ptr) % align));
    return 0;
  };
  auto AHost = [&](size_t align, auto... args) {
    auto ptr = aligned_alloc_host(align, N, args...);
    assert(!ptr || !(reinterpret_cast<std::uintptr_t>(ptr) % align));
    return 0;
  };
  auto AShared = [&](size_t align, auto... args) {
    void *ptr = nullptr;
    if (dev.has(aspect::usm_shared_allocations)) {
      ptr = aligned_alloc_shared(align, N, args...);
    }
    assert(!ptr || !(reinterpret_cast<std::uintptr_t>(ptr) % align));
    return 0;
  };
  auto AAnnotated = [&](size_t align, auto... args) {
    auto ptr = aligned_alloc(align, N, args...);
    assert(!ptr || !(reinterpret_cast<std::uintptr_t>(ptr) % align));
    return 0;
  };

  auto CheckNullOrAlignedAll = [&](auto Funcs) {
    std::apply([&](auto... Fs) { (void)std::initializer_list<int>{Fs()...}; },
               Funcs);
  };

  CheckNullOrAlignedAll(std::tuple{
      [&]() { return ADevice(3, q); }, [&]() { return ADevice(5, dev, Ctx); },
      [&]() { return AHost(7, q); }, [&]() { return AHost(9, Ctx); },
      [&]() { return AShared(114, q); },
      [&]() { return AShared(1023, dev, Ctx); },
      [&]() { return AAnnotated(15, q, alloc::device); },
      [&]() { return AAnnotated(17, dev, Ctx, alloc::host); }});
}

int main() {
  sycl::queue q;
  testAlign<char>(q, 4);
  testAlign<int>(q, 128);
  testAlign<std::complex<double>>(q, 4);
  return 0;
}
