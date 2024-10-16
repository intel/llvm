// XFAIL: native_cpu
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>

#include <iomanip>

using namespace sycl;
namespace sycl_exp = sycl::ext::oneapi::experimental;

template <typename T> void test() {
  std::cout << std::setw(50) << __PRETTY_FUNCTION__ << ", time:";
  constexpr int WG_SIZE = 32 + 16 + 8 + 4;
  constexpr int GLOBAL_SIZE = WG_SIZE * 1;

  queue q;

  buffer<T, 1> b{GLOBAL_SIZE};

  for (int i = 0; i < 5; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&](handler &cgh) {
       accessor acc{b, cgh};
       size_t temp_memory_size = WG_SIZE * sizeof(T);
       auto scratch = sycl::local_accessor<std::byte, 1>(temp_memory_size, cgh);

       cgh.parallel_for(
           nd_range<1>{range<1>{GLOBAL_SIZE}, range<1>{WG_SIZE}},
           [=](nd_item<1> ndi) {
             auto g = ndi.get_group();
             auto sg = ndi.get_sub_group();
             // sg's scratch space starts at sg leader's *group* linear id.
             auto sg_scratch = sycl::span(
                 &scratch[group_broadcast(sg, g.get_local_linear_id())],
                 sizeof(T) * sg.get_local_linear_range());
             auto handle = sycl_exp::group_with_scratchpad(sg, sg_scratch);
             T val{0};
             auto binop = [](T x, T y) { return x + y; };
             for (int j = 0; j < 100000; ++j)
               val += sycl_exp::reduce_over_group(
                   handle, static_cast<T>(j % 100), binop);
             acc[ndi.get_global_linear_id()] = val;
           });
     }).wait();
    if (i == 0)
      continue; // skip first iteration's overheads.
    auto end = std::chrono::high_resolution_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count();
    std::cout << " " << std::setw(6) << time << "ms";
  }
  std::cout << std::endl;
}

int main() {
  test<uint8_t>();
  test<uint16_t>();
  test<uint32_t>();
  test<uint64_t>();

  test<int8_t>();
  test<int16_t>();
  test<int32_t>();
  test<int64_t>();

  if (device{}.has(aspect::fp16))
    test<half>();
  test<float>();
  if (device{}.has(aspect::fp64))
    test<double>();

  return 0;
}
