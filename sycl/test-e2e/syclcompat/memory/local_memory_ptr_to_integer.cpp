// REQUIRES: cuda
// RUN:  %{build} -Xsycl-target-backend --cuda-gpu-arch=sm_75 -o %t.out
// RUN:  %{run} %t.out
#include <sycl/detail/core.hpp>
#include <sycl/group_barrier.hpp>
#include <syclcompat/memory.hpp>

using namespace sycl;
#define NUM_ELEMENTS 64

template <class T> void test(queue stream) {
  half *res = malloc_shared<half>(NUM_ELEMENTS, stream);

  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    res[i] = 0.5;
  }

  sycl::nd_range<1> global_range{sycl::range{32}, sycl::range{32}};

  stream
      .submit([&](handler &h) {
        h.parallel_for<T>(global_range, [=](nd_item<1> item) {
          sycl::group work_group = item.get_group();
          int id = item.get_global_linear_id();
          half *data = syclcompat::local_mem<half[NUM_ELEMENTS]>();

          data[id * 2] = id;
          data[id * 2 + 1] = id + 0.5;

          T addr =
              syclcompat::ptr_to_int<T>(reinterpret_cast<char *>(data) + (id % 8) * 16);

          uint32_t fragment;
#if defined(__NVPTX__)
          asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                       : "=r"(fragment)
                       : "r"(addr));
#endif
          sycl::group_barrier(work_group);

          half *data_ptr = reinterpret_cast<half *>(&fragment);
          res[id * 2] = data_ptr[0];
          res[id * 2 + 1] = data_ptr[1];
        });
      })
      .wait();

  for (int i = 0; i < NUM_ELEMENTS; i++) {
    assert(res[i] == static_cast<half>(i / 2.0));
  }

  free(res, stream);
};

int main() {

  queue stream{property::queue::in_order{}};
  test<size_t>(stream);
  test<uint32_t>(stream);

  std::cout << "PASS" << std::endl;
  return 0;
}
