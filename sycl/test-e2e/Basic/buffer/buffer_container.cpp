// RUN: %clangxx %cxx_std_optionc++17 -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out
// RUN: %GPU_RUN_PLACEHOLDER %t2.out
// RUN: %ACC_RUN_PLACEHOLDER %t2.out

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  // buffer created from contiguous container copies back
  {
    constexpr int num = 10;
    std::vector<int> out_data(num, -1);
    {
      buffer A(out_data);
      queue Q;
      Q.submit([&](handler &h) {
        auto out = A.get_access<access::mode::write>(h);
        h.parallel_for<class containerBuffer>(A.get_range(),
                                              [out](id<1> i) { out[i] = 1; });
      });
    } //~buffer
    for (int i = 0; i < num; i++)
      assert(out_data[i] == 1);
  }
}
