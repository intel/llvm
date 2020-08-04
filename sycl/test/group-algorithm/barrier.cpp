// UNSUPPORTED: cuda
//
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <cassert>
using namespace sycl;
using namespace sycl::intel;

class barrier_kernel;

void test(queue q) {

  constexpr size_t N = 32;
  constexpr size_t L = 16;
  std::array<int, N> out;
  std::fill(out.begin(), out.end(), 0);
  {
    buffer<int> out_buf(out.data(), range<1>{N});
    q.submit([&](handler &cgh) {
      auto tmp =
          accessor<int, 1, access::mode::read_write, access::target::local>(
              L, cgh);
      auto out = out_buf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class barrier_kernel>(
          nd_range<1>(N, L), [=](nd_item<1> it) {
            group<1> g = it.get_group();
            tmp[it.get_local_linear_id()] = it.get_global_linear_id() + 1;
            barrier(g);
            int result = 0;
            for (int i = 0; i < L; ++i) {
              result += tmp[i];
            }
            out[it.get_global_linear_id()] = result;
          });
    });
  }

  // Each work-item should see writes from all other work-items in its group
  for (int g = 0; g < N / L; ++g) {
    int sum = 0;
    for (int wi = 0; wi < L; ++wi) {
      sum += g * L + wi + 1;
    }
    for (int wi = 0; wi < L; ++wi) {
      assert(out[g * L + wi] == sum);
    }
  }
}

int main() {
  queue q;
  test(q);
  std::cout << "Test passed." << std::endl;
}
