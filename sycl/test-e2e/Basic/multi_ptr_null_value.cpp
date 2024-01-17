// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// REQUIRES: cuda || hip || level_zero

#include <sycl/sycl.hpp>

template <sycl::access::address_space address_space> void nullPtrTest() {
  auto queue = sycl::queue();
  int value = 0;
  {
    sycl::buffer<int> val_buffer(&value, sycl::range(1));
    queue
        .submit([&](sycl::handler &cgh) {
          auto acc_for_multi_ptr = val_buffer.template get_access(cgh);
          cgh.single_task([=] {
            sycl::multi_ptr<int, address_space, sycl::access::decorated::yes>
                mp;
            mp = nullptr;
            acc_for_multi_ptr[0] = mp.get_raw() == nullptr;
          });
        })
        .wait_and_throw();
  }

  assert(value && "Invalid value for multi_ptr nullptr comparison!");
}

int main() {
  nullPtrTest<sycl::access::address_space::local_space>();
  nullPtrTest<sycl::access::address_space::private_space>();
  nullPtrTest<sycl::access::address_space::global_space>();
  nullPtrTest<sycl::access::address_space::generic_space>();
  return 0;
}
