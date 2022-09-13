// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <vector>

using namespace sycl;

int main() {
  std::vector<int> reference = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

  {
    buffer<int, 1> buf(reference.data(), range<1>{10});
    auto acc = buf.get_access<access_mode::read_write>(range<1>{10});
    std::vector<int> data;
    auto It = acc.begin();
    std::cout << *(It--) << std::endl;
    std::cout << *(It--) << std::endl;
    std::cout << *(--It) << std::endl;
    std::cout << *(--It) << std::endl;
    /*int N = 0;
    for (auto I = acc.begin(), E = acc.end(); I != E; ++I) {
      data.push_back(*I);
      std::cout << *I << std::endl;
      ++N;
      if (N > 20)
        break;
    }*/

    // assert
  }

  return 0;
}
