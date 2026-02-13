// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// This test verifies that we can use scoped enum types as arguments in free
// function kernels.

#include "free_function_user_enum_class.hpp"
#include "helpers.hpp"
#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/ext/oneapi/work_group_static.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;
namespace at::native::xpu {

enum class OP_MODE_SCOPED : uint8_t { INC, DEC, MUL, DIV };
enum OP_MODE : uint8_t { INC, DEC, MUL, DIV };

template <typename T, ADAM_MODE adam_mode, int SIMD> struct LpMaxFunctor {
  void operator()(sycl::nd_item<1> item, T *data, size_t size) {
    size_t idx = item.get_global_linear_id();
    if (idx < size) {
      if constexpr (adam_mode == ADAM_MODE::ADAMW)
        data[idx] += 1;
      else
        data[idx] += 2;
    }
  }
};

template <typename T, OP_MODE op_mode> struct OpFunctor {
  void operator()(sycl::nd_item<1> item, T *data, size_t size) {
    size_t idx = item.get_global_linear_id();
    if (idx < size) {
      if constexpr (op_mode == OP_MODE::DEC)
        data[idx]--;
    }
  }
};

template <typename T, OP_MODE_SCOPED op_mode> struct OpSFunctor {
  void operator()(sycl::nd_item<1> item, T *data, size_t size) {
    size_t idx = item.get_global_linear_id();
    if (idx < size) {
      if constexpr (op_mode == OP_MODE_SCOPED::INC)
        data[idx]++;
      else if constexpr (op_mode == OP_MODE_SCOPED::DEC)
        data[idx]--;
      else if constexpr (op_mode == OP_MODE_SCOPED::MUL)
        data[idx] *= 2;
      else if constexpr (op_mode == OP_MODE_SCOPED::DIV)
        data[idx] /= 2;
    }
  }
};

template <typename U, typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void applyKernel(U callable, T *data, size_t size) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  callable(item, data, size);
}

template <auto *kptr, int dim, typename... Kargs>
static inline void sycl_kernel_submit(::sycl::range<dim> global_range,
                                      ::sycl::range<dim> local_range,
                                      ::sycl::queue q, int slm_sz,
                                      Kargs... args) {
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  if (slm_sz != 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<dim>(::sycl::range<dim>(global_range),
                              ::sycl::range<dim>(local_range)),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, ker, args...);
  } else {
    syclexp::launch_config cfg{::sycl::nd_range<dim>(
        ::sycl::range<dim>(global_range), ::sycl::range<dim>(local_range))};
    syclexp::nd_launch(q, cfg, ker, args...);
  }
}

template <typename T>
void apply_lp(sycl::queue &q, T callable, int *data, size_t size,
              size_t global_size) {
  constexpr auto kernel =
      applyKernel<LpMaxFunctor<int, ADAM_MODE::ADAMW, 32>, int>;
  sycl_kernel_submit<kernel>(sycl::range<1>(global_size), sycl::range<1>(1), q,
                             0, callable, data, size);
}

template <typename T>
void apply_op_scoped(sycl::queue &q, T callable, int *data, size_t size,
                     size_t global_size) {
  constexpr auto kernel =
      applyKernel<OpSFunctor<int, OP_MODE_SCOPED::MUL>, int>;
  sycl_kernel_submit<kernel>(sycl::range<1>(global_size), sycl::range<1>(1), q,
                             0, callable, data, size);
}

template <typename T>
void apply_op(sycl::queue &q, T callable, int *data, size_t size,
              size_t global_size) {
  constexpr auto kernel = applyKernel<OpFunctor<int, OP_MODE::DEC>, int>;
  sycl_kernel_submit<kernel>(sycl::range<1>(global_size), sycl::range<1>(1), q,
                             0, callable, data, size);
}

} // namespace at::native::xpu

int main() {
  sycl::queue q;
  constexpr size_t N = 10;
  auto *data = sycl::malloc_shared<int>(N, q);
  assert(data && "USM allocation failed");

  for (size_t i = 0; i < N; ++i)
    data[i] = i + 1;

  at::native::xpu::LpMaxFunctor<int, at::native::xpu::ADAM_MODE::ADAMW, 32>
      subKernel;
  apply_lp(q, subKernel, data, N, N);
  q.wait_and_throw();

  for (size_t i = 0; i < N; ++i) {
    assert(data[i] == i + 2);
    data[i] = i;
  }

  at::native::xpu::OpSFunctor<int, at::native::xpu::OP_MODE_SCOPED::MUL>
      subKernelOpS;
  apply_op_scoped(q, subKernelOpS, data, N, N);
  q.wait_and_throw();

  for (size_t i = 0; i < N; ++i) {
    assert(data[i] == i * 2);
    data[i] = i;
  }

  at::native::xpu::OpFunctor<int, at::native::xpu::OP_MODE::DEC> subKernelOp;
  apply_op(q, subKernelOp, data, N, N);
  q.wait_and_throw();

  for (size_t i = 0; i < N; ++i) {
    assert(data[i] == (i - 1) && "DEC operation failed");
    data[i] = i;
  }

  sycl::free(data, q);
  return 0;
}
