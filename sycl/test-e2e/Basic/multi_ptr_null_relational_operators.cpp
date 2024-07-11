// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>
#include <sycl/multi_ptr.hpp>

template <typename T, typename AccessorType,
          sycl::access::address_space address_space,
          sycl::access::decorated decorated>
void check(sycl::multi_ptr<T, address_space, decorated> &mp,
           AccessorType &dev_acc) {
  using multi_ptr_t = sycl::multi_ptr<T, address_space, decorated>;
  multi_ptr_t null_mp;
  dev_acc[0] = nullptr == null_mp;
  dev_acc[0] += nullptr != mp;
  dev_acc[0] += std::less<multi_ptr_t>()(nullptr, mp) == nullptr < mp;
  dev_acc[0] += std::less<multi_ptr_t>()(mp, nullptr) == mp < nullptr;
  dev_acc[0] += std::less_equal<multi_ptr_t>()(nullptr, mp) == nullptr <= mp;
  dev_acc[0] += std::less_equal<multi_ptr_t>()(mp, nullptr) == mp <= nullptr;
  dev_acc[0] += std::greater<multi_ptr_t>()(nullptr, mp) == nullptr > mp;
  dev_acc[0] += std::greater<multi_ptr_t>()(mp, nullptr) == mp > nullptr;
  dev_acc[0] += std::greater_equal<multi_ptr_t>()(nullptr, mp) == nullptr >= mp;
  dev_acc[0] += std::greater_equal<multi_ptr_t>()(mp, nullptr) == mp >= nullptr;
}

template <typename T, sycl::access::address_space address_space,
          sycl::access::decorated decorated>
void nullptrRelationalOperatorTest() {
  using multi_ptr_t = sycl::multi_ptr<int, address_space, decorated>;
  try {
    sycl::queue queue;
    sycl::buffer<int, 1> buf(1);
    queue
        .submit([&](sycl::handler &cgh) {
          auto dev_acc = buf.get_access<sycl::access::mode::write>(cgh);
          if constexpr (address_space ==
                        sycl::access::address_space::local_space) {
            sycl::local_accessor<int, 1> locAcc(1, cgh);
            cgh.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
              locAcc[0] = 1;
              multi_ptr_t mp(locAcc);
              check(mp, dev_acc);
            });
          } else if constexpr (address_space ==
                               sycl::access::address_space::private_space) {
            cgh.single_task([=] {
              T priv_arr[1];
              sycl::multi_ptr<T, address_space, decorated> mp =
                  sycl::address_space_cast<address_space, decorated>(priv_arr);
              check(mp, dev_acc);
            });
          } else {
            cgh.single_task([=] {
              multi_ptr_t mp(dev_acc);
              check(mp, dev_acc);
            });
          }
        })
        .wait_and_throw();
    assert(sycl::host_accessor{buf}[0] == 10);
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return;
  }
}

int main() {
  nullptrRelationalOperatorTest<int, sycl::access::address_space::local_space,
                                sycl::access::decorated::yes>();
  nullptrRelationalOperatorTest<int, sycl::access::address_space::local_space,
                                sycl::access::decorated::no>();
  nullptrRelationalOperatorTest<int, sycl::access::address_space::global_space,
                                sycl::access::decorated::yes>();
  nullptrRelationalOperatorTest<int, sycl::access::address_space::global_space,
                                sycl::access::decorated::no>();
  nullptrRelationalOperatorTest<int, sycl::access::address_space::generic_space,
                                sycl::access::decorated::yes>();
  nullptrRelationalOperatorTest<int, sycl::access::address_space::generic_space,
                                sycl::access::decorated::no>();
  nullptrRelationalOperatorTest<int, sycl::access::address_space::private_space,
                                sycl::access::decorated::yes>();
  nullptrRelationalOperatorTest<int, sycl::access::address_space::private_space,
                                sycl::access::decorated::no>();
  return 0;
}
