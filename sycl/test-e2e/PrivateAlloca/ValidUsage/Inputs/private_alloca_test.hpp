#pragma once

// Template for private alloca tests.

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>
#include <sycl/specialization_id.hpp>

template <typename ElementType, typename SizeType,
          sycl::access::decorated DecorateAddress, std::size_t Alignment>
class Kernel;

template <typename ElementType, auto &Size,
          sycl::access::decorated DecorateAddress, std::size_t Alignment = 0>
void test(std::size_t n) {
  std::vector<std::size_t> v(n);
  {
    sycl::queue q;
    sycl::buffer<std::size_t> b(v);
    q.submit([&](sycl::handler &cgh) {
      sycl::accessor acc(b, cgh, sycl::write_only, sycl::no_init);
      cgh.set_specialization_constant<Size>(n);
      using spec_const_type = std::remove_reference_t<decltype(Size)>;
      using size_type = typename spec_const_type::value_type;
      cgh.single_task<
          Kernel<ElementType, size_type, DecorateAddress, Alignment>>(
          [=](sycl::kernel_handler h) {
            sycl::private_ptr<ElementType, DecorateAddress> ptr;
            if constexpr (Alignment > 0) {
              ptr = sycl::ext::oneapi::experimental::aligned_private_alloca<
                  ElementType, Alignment, Size, DecorateAddress>(h);
            } else {
              ptr = sycl::ext::oneapi::experimental::private_alloca<
                  ElementType, Size, DecorateAddress>(h);
            }
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
  assert(static_cast<std::size_t>(v.front()) == n &&
         "Wrong private alloca length reported");
  for (std::size_t i = 1; i < n; ++i) {
    assert(static_cast<std::size_t>(v[i]) == i &&
           "Wrong value in copied-back sequence");
  }
}
