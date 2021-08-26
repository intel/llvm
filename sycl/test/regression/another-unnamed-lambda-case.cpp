// RUN: %clangxx -fsycl -fsycl-unnamed-lambda %s -o %t.out
#include <sycl.hpp>
int main()
{
  auto w = [](auto i){};
    sycl::queue q;
    q.parallel_for(10, [](auto i){});
    q.parallel_for(10, w);
}
