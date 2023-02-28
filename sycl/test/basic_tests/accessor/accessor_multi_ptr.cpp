// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only

#include <cassert>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl;

constexpr static int size = 1;

void test_get_multi_ptr(handler &cgh, buffer<int, size> &buffer) {
  using accessor_t =
      accessor<int, size, access::mode::read_write, access::target::host_task,
               access::placeholder::true_t>;
  using local_accessor_t = local_accessor<int, size>;

  auto ptr = buffer.get_host_access(cgh);
  auto local_ptr = local_accessor<int, size>({size}, cgh);
  auto acc_ptr = ptr.get_pointer();
  auto acc_multi_ptr = ptr.get_multi_ptr<access::decorated::yes>();
  auto local_acc_multi_ptr = local_ptr.get_multi_ptr<access::decorated::yes>();
  auto local_acc_ptr = local_ptr.get_pointer();

  static_assert(std::is_same_v<decltype(acc_ptr), std::add_pointer_t<int>>);

  static_assert(
      std::is_same_v<decltype(local_acc_ptr), std::add_pointer_t<int>>);

  static_assert(
      std::is_same_v<
          decltype(acc_multi_ptr),
          typename accessor_t::template accessor_ptr<access::decorated::yes>>);

  static_assert(std::is_same_v<decltype(local_acc_multi_ptr),
                               typename local_accessor_t::template accessor_ptr<
                                   access::decorated::yes>>);
}
