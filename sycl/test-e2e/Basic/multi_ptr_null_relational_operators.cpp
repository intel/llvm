// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/sycl.hpp>

template <typename multi_ptr_t> void nullptrRelationalOperatorTest() {
  constexpr int OUTPUT_SIZE = 8;
  bool output[OUTPUT_SIZE];
  try {
    sycl::queue queue;
    sycl::buffer<bool, 1> buf(output, sycl::range<1>(OUTPUT_SIZE));
    queue
        .submit([&](sycl::handler &cgh) {
          auto dev_acc = buf.get_access<sycl::access::mode::write>(cgh);
          sycl::local_accessor<bool, 1> locAcc(1, cgh);
          cgh.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::id<1>) {
            locAcc[0] = 10;
            multi_ptr_t mp(locAcc);
            dev_acc[0] = std::less<multi_ptr_t>()(nullptr, mp) == nullptr < mp;
            dev_acc[1] = std::less<multi_ptr_t>()(mp, nullptr) == mp < nullptr;
            dev_acc[2] =
                std::less_equal<multi_ptr_t>()(nullptr, mp) == nullptr <= mp;
            dev_acc[3] =
                std::less_equal<multi_ptr_t>()(mp, nullptr) == mp <= nullptr;
            dev_acc[4] =
                std::greater<multi_ptr_t>()(nullptr, mp) == nullptr > mp;
            dev_acc[5] =
                std::greater<multi_ptr_t>()(mp, nullptr) == mp > nullptr;
            dev_acc[6] =
                std::greater_equal<multi_ptr_t>()(nullptr, mp) == nullptr >= mp;
            dev_acc[7] =
                std::greater_equal<multi_ptr_t>()(mp, nullptr) == mp >= nullptr;
          });
        })
        .wait_and_throw();
  } catch (sycl::exception e) {
    std::cout << "SYCL exception caught: " << e.what();
    return;
  }
  for (int index = 0; index < OUTPUT_SIZE; ++index) {
    assert(output[index] && "Unexpected relational operator result.");
  }
}

int main() {
  using multi_ptr_yes =
      sycl::multi_ptr<bool, sycl::access::address_space::local_space,
                      sycl::access::decorated::yes>;
  using multi_ptr_no =
      sycl::multi_ptr<bool, sycl::access::address_space::local_space,
                      sycl::access::decorated::no>;
  nullptrRelationalOperatorTest<multi_ptr_yes>();
  nullptrRelationalOperatorTest<multi_ptr_no>();
  return 0;
}
