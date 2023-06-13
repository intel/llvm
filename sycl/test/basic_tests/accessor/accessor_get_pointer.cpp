// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -fsyntax-only

#include <cassert>
#include <sycl/sycl.hpp>
#include <type_traits>

using namespace sycl;

constexpr static int size = 1;

void test_get_multi_ptr(handler &cgh, buffer<int, size> &buffer) {
  using target_local_accessor_t =
      accessor<int, size, access::mode::read_write, access::target::local>;
  using local_accessor_t = local_accessor<int, size>;
  using accessor_t =
      accessor<int, size, access::mode::read_write, access::target::device>;

  auto acc = buffer.get_access<access_mode::read_write, target::host_task>(cgh);
  auto target_local_acc = target_local_accessor_t({size}, cgh);
  auto local_acc = local_accessor_t({size}, cgh);
  auto device_acc =
      buffer.get_access<access_mode::read_write, target::device>(cgh);

  auto acc_ptr = acc.get_pointer();
  auto target_local_ptr = target_local_acc.get_pointer();
  auto local_pointer = local_acc.get_pointer();
  auto device_acc_ptr = device_acc.get_pointer();
  static_assert(std::is_same_v<decltype(acc_ptr), std::add_pointer_t<int>>);
  static_assert(std::is_same_v<decltype(target_local_ptr), local_ptr<int>>);
  static_assert(
      std::is_same_v<decltype(local_pointer), std::add_pointer_t<int>>);
  static_assert(
      std::is_same_v<decltype(device_acc_ptr), std::add_pointer_t<int>>);
}
