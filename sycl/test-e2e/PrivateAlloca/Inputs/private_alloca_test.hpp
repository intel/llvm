#pragma once

// Template for private alloca tests.

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>

template <typename ElementType, typename SizeType,
          sycl::access::decorated DecorateAddress>
class Kernel;

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress>
void test() {
  std::size_t N;

  std::cin >> N;

  std::vector<std::size_t> v(N);
  {
    sycl::queue q;
    sycl::buffer<std::size_t> b(v);
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc(b, cgh, sycl::write_only, sycl::no_init);
      cgh.set_specialization_constant<Size>(N);
      using spec_const_type = std::remove_reference_t<decltype(Size)>;
      using size_type = typename spec_const_type::value_type;
      cgh.single_task<Kernel<ElementType, size_type, DecorateAddress>>(
          [=](sycl::kernel_handler h) {
            auto ptr = sycl::ext::oneapi::experimental::private_alloca<
                ElementType, Size, DecorateAddress>(h);
            const std::size_t M = h.get_specialization_constant<Size>();
            ptr[0] = static_cast<ElementType>(M);
            ElementType value{1};
            for (auto begin = ptr.get() + 1, end = ptr.get() + M; begin < end;
                 ++begin, ++value) {
              *begin = value;
            }
            auto accBegin = acc.begin();
            for (auto begin = ptr.get(), end = ptr.get() + M; begin < end;
                 ++begin, ++accBegin) {
              *accBegin = *begin;
            }
          });
    });
    q.wait_and_throw();
  }
  assert(static_cast<std::size_t>(v.front()) == N &&
         "Wrong private alloca length reported");
  for (std::size_t i = 1; i < N; ++i) {
    assert(static_cast<std::size_t>(v[i]) == i &&
           "Wrong value in copied-back sequence");
  }
}
