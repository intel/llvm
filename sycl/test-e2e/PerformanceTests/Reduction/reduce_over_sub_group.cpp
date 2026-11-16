// XFAIL: native_cpu, gpu-intel-gen12
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/16415
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/17872
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// PERF-DESCRIPTION: Time of 100000 sub-group reduce_over_group per type.

#include "../perf_harness.hpp"

#include <sycl/detail/core.hpp>
#include <sycl/sub_group.hpp>

#include <sycl/ext/oneapi/experimental/user_defined_reductions.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <type_traits>

using namespace sycl;
namespace sycl_exp = sycl::ext::oneapi::experimental;

constexpr int WG_SIZE = 32 + 16 + 8 + 4;
constexpr int GLOBAL_SIZE = WG_SIZE * 1;
constexpr int ITERS = 100000;

template <typename T> void add_case(const char *TypeName) {
  perf::add_case(
      std::string("reduce_over_group sub-group ") + TypeName, "ms",
      [](perf::State &S) {
        queue q;
        buffer<T, 1> b{GLOBAL_SIZE};
        buffer<uint32_t, 1> sg_sizes{GLOBAL_SIZE};

        S.measure([&] {
          q.submit([&](handler &cgh) {
             accessor acc{b, cgh};
             accessor sg_size_acc{sg_sizes, cgh};
             size_t temp_memory_size = WG_SIZE * sizeof(T);
             auto scratch =
                 sycl::local_accessor<std::byte, 1>(temp_memory_size, cgh);

             cgh.parallel_for(
                 nd_range<1>{range<1>{GLOBAL_SIZE}, range<1>{WG_SIZE}},
                 [=](nd_item<1> ndi) {
                   auto g = ndi.get_group();
                   auto sg = ndi.get_sub_group();
                   // sg's scratch space starts at sg leader's *group* linear
                   // id, in units of T.
                   auto sg_scratch = sycl::span(
                       &scratch[sizeof(T) *
                                group_broadcast(sg, g.get_local_linear_id())],
                       sizeof(T) * sg.get_local_linear_range());
                   auto handle =
                       sycl_exp::group_with_scratchpad(sg, sg_scratch);
                   T val{0};
                   auto binop = [](T x, T y) { return x + y; };
                   for (int j = 0; j < ITERS; ++j)
                     val += sycl_exp::reduce_over_group(
                         handle, static_cast<T>(j % 100), binop);
                   acc[ndi.get_global_linear_id()] = val;
                   sg_size_acc[ndi.get_global_linear_id()] =
                       sg.get_local_linear_range();
                 });
           }).wait();
        });

        // half accumulates past its range, so only integer and float/double
        // results are compared against a host-computed value.
        if constexpr (!std::is_same_v<T, half>) {
          host_accessor res{b};
          host_accessor sg_size_res{sg_sizes};
          for (int i = 0; i < GLOBAL_SIZE; ++i) {
            uint64_t SgSize = sg_size_res[i];
            T Expected{0};
            if constexpr (std::is_integral_v<T>) {
              uint64_t Sum = 0;
              for (int j = 0; j < ITERS; ++j)
                Sum += SgSize * (j % 100);
              Expected = static_cast<T>(Sum);
            } else {
              for (int j = 0; j < ITERS; ++j)
                Expected += static_cast<T>(SgSize * (j % 100));
            }
            perf::check(res[i] == Expected, "reduction result");
          }
        }
      },
      perf::Options{1, 5});
}

int main(int argc, char **argv) {
  add_case<uint8_t>("uint8");
  add_case<uint16_t>("uint16");
  add_case<uint32_t>("uint32");
  add_case<uint64_t>("uint64");

  add_case<int8_t>("int8");
  add_case<int16_t>("int16");
  add_case<int32_t>("int32");
  add_case<int64_t>("int64");

  if (device{}.has(aspect::fp16))
    add_case<half>("half");
  add_case<float>("float");
  if (device{}.has(aspect::fp64))
    add_case<double>("double");

  return perf::main(argc, argv);
}
