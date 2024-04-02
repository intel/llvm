// REQUIRES: cuda
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// XFAIL: *

#include <numeric>
#include <string_view>
#include <type_traits>

#include <sycl/sycl.hpp>

// The test aims to use more than 64 registers meaning that with the maximum
// workgroup size (assumed 1024 on most CUDA SMs), will produce a launch
// configuration that will execution due to reaching HW limitations. Hence, we
// are able to check the functionality of the out-of-registers (exceeded
// max-registers-per-block) error handling in the SYCL runtime.
//
// A more reliable test to work with lower max work-group sizes will be better,
// but a also more complicated to achieve the register usage desired to ensure,
// we will reach the hardware limitations. Additionally, all GPUs supported by
// CUDA backend allow max block sizes of 1024, hence it is safe to rely on that
// assumption.
//
// To avoid false test failure, we early exit if the said assumption isn't met.

class kernel_vadd_and_sum;

int main() {
  sycl::queue q;
  sycl::device dev = q.get_device();
  size_t local_size = dev.get_info<sycl::info::device::max_work_group_size>();
  if (local_size < 1024u) {
    // Skip because we may not reach the error case due to the register count.
    return 0;
  }

  using elem_t = unsigned int;
  static_assert(std::is_integral_v<elem_t> || std::is_floating_point_v<elem_t>);

  constexpr int VEC_DIM = 16;
  static_assert(sycl::detail::isValidVectorSize(VEC_DIM));

  constexpr size_t GLOBAL_WORK_SIZE = 1024u * VEC_DIM;

  sycl::buffer<sycl::vec<elem_t, VEC_DIM>> valuesBuf1{GLOBAL_WORK_SIZE};
  sycl::buffer<sycl::vec<elem_t, VEC_DIM>> valuesBuf2{GLOBAL_WORK_SIZE};
  sycl::buffer<sycl::vec<elem_t, VEC_DIM>> valuesBuf3{GLOBAL_WORK_SIZE};
  sycl::buffer<sycl::vec<elem_t, VEC_DIM>> valuesBuf4{GLOBAL_WORK_SIZE};
  {
    sycl::host_accessor a1{valuesBuf1};
    std::iota(a1.begin(), a1.end(), sycl::vec<elem_t, VEC_DIM>(elem_t{0}));

    sycl::host_accessor a2{valuesBuf2};
    std::iota(a2.begin(), a2.end(), sycl::vec<elem_t, VEC_DIM>(elem_t{0}));

    sycl::host_accessor a3{valuesBuf3};
    std::iota(a3.begin(), a3.end(), sycl::vec<elem_t, VEC_DIM>(elem_t{0}));

    sycl::host_accessor a4{valuesBuf4};
    std::iota(a4.begin(), a4.end(), sycl::vec<elem_t, VEC_DIM>(elem_t{0}));
  }

  sycl::buffer<sycl::vec<elem_t, VEC_DIM>> outputBuf{GLOBAL_WORK_SIZE};

  try {
    q.submit([&](sycl::handler &h) {
       auto input1 = valuesBuf1.get_access<sycl::access::mode::read>(h);
       auto input2 = valuesBuf2.get_access<sycl::access::mode::read>(h);
       auto input3 = valuesBuf3.get_access<sycl::access::mode::read>(h);
       auto input4 = valuesBuf4.get_access<sycl::access::mode::read>(h);
       auto output = outputBuf.get_access<sycl::access::mode::write>(h);
       h.parallel_for<kernel_vadd_and_sum>(
           sycl::nd_range<1>{{GLOBAL_WORK_SIZE}, {local_size}},
           [=](sycl::id<1> i) {
             sycl::vec<elem_t, VEC_DIM> values1 = input1[i];
             sycl::vec<elem_t, VEC_DIM> values2 = input2[i];
             sycl::vec<elem_t, VEC_DIM> values3 = input3[i];
             sycl::vec<elem_t, VEC_DIM> values4 = input4[i];

             // compute vector add
             const auto vadd = values1 + values2 + values3 + values4;

             // compute total vector elements sum
             auto sum = elem_t(0);
             for (int j = 0; j < VEC_DIM; j++) {
               sum += input1[i][j];
               sum += input2[i][j];
               sum += input3[i][j];
               sum += input4[i][j];
             }

             output[i] = vadd;
             output[i] += sum;
           });
     }).wait();
  } catch (sycl::exception &e) {
    using std::string_view_literals::operator""sv;
    auto Msg = "Exceeded the number of registers available on the hardware."sv;
    if (std::string(e.what()).find(Msg) != std::string::npos) {
      return 0;
    }
  }

  return 1;
}
