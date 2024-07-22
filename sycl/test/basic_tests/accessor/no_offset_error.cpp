// RUN:  %clangxx -fsycl-device-only -Xclang -verify -Xclang -verify-ignore-unexpected=note -emit-llvm -o - %s

#include <sycl/sycl.hpp>

inline constexpr int size = 100;

int main() {

  sycl::buffer<int> a{sycl::range{size}};
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    sycl::ext::oneapi::accessor_property_list PL{sycl::ext::oneapi::no_offset,
                                                 sycl::no_init};
    sycl::accessor acc_a(a, cgh, sycl::write_only, PL);
    // expected-error@sycl/accessor.hpp:* {{static assertion failed due to requirement '!(accessor_property_list<sycl::ext::oneapi::property::no_offset::instance<true>, sycl::property::no_init>::has_property())': Accessor has no_offset property, get_offset() can not be used}}
    auto b = acc_a.get_offset();
  });

  q.wait();
  return 0;
}
