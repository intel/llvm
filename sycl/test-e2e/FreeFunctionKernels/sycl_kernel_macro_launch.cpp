// REQUIRES: aspect-usm_shared_allocations
// UNSUPPORTED: target-amd
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/16072

// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// XFAIL: target-native_cpu
// XFAIL-TRACKER: https://github.com/intel/llvm/issues/20142

// CUDA-`<<<>>>`-style bare-name launch via the SYCL_EXT_ONEAPI_KERNEL_FUNCTION macro. The existing
// enqueue-function API (nd_launch / single_task taking a kernel_function<Func>
// selector) is kept unchanged; SYCL_EXT_ONEAPI_KERNEL_FUNCTION(name, args...) expands to
//   kernel_function<__builtin_sycl_launch_kernel(name, args...)>, args...
// so the compiler deduces the kernel's template arguments / resolves the
// overload from the launch arguments while the launched SPIR-V kernel remains
// the real user function (no wrapper). This test exercises single_task and
// nd_launch; non-templated and templated kernels; a dependent (function-
// template) call site; a zero-argument kernel; a kernel with mixed template
// parameters (an explicit non-deducible Dim plus a deduced T); an overload set
// resolved from the launch argument types; and every launch-configuration form
// the macro can target — queue and handler overloads, and the launch_config
// overload.

#include <cassert>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/oneapi/experimental/free_function_traits.hpp>
#include <sycl/ext/oneapi/free_function_queries.hpp>
#include <sycl/usm.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

// Non-templated single_task.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void store42(int *p) { *p = 42; }

// Templated single_task.
template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void store_one(T *p) { *p = T{1}; }

// Non-templated nd_range.
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void scale(float *y, float k, int n) {
  size_t i = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if (static_cast<int>(i) < n)
    y[i] *= k;
}

// Templated nd_range.
template <typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void axpy(T *y, const T *x, T a, int n) {
  size_t i = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if (static_cast<int>(i) < n)
    y[i] = a * x[i] + y[i];
}

// Zero-argument single_task kernel (no launch args after the name).
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::single_task_kernel))
void tick() {}

// Mixed template parameters: Dim is a non-deducible NTTP given explicitly at
// the launch site; T is deduced from the launch arguments. The kernel-kind
// property is itself parameterized on Dim.
template <int Dim, typename T>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<Dim>))
void fill_val(T *p, T v, int n) {
  size_t i = syclext::this_work_item::get_nd_item<Dim>().get_global_linear_id();
  if (static_cast<int>(i) < n)
    p[i] = v;
}

// Overload set: two free function kernels of the same name distinguished by
// argument type. The builtin resolves which overload from the launch arguments.
// (This required intel/llvm#22793: launching a free function kernel directly
// instead of through a wrapper — the wrapper's integration-header entry could
// not represent an overloaded name.)
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void fill(int *p, int v, int n) {
  size_t i = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if (static_cast<int>(i) < n)
    p[i] = v;
}
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void fill(float *p, float v, int n) {
  size_t i = syclext::this_work_item::get_nd_item<1>().get_global_linear_id();
  if (static_cast<int>(i) < n)
    p[i] = v;
}

// A SYCL_EXT_ONEAPI_KERNEL_FUNCTION launch from inside an ordinary function template, so the launch
// arguments are dependent and the macro's builtin call is deferred to
// instantiation (regression guard: this dependent-context use previously
// crashed the front end before the builtin call was made type-dependent).
template <typename T>
void run_axpy(sycl::queue q, sycl::nd_range<1> r, T *y, const T *x, T a, int n) {
  syclexp::nd_launch(q, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(axpy, y, x, a, n));
}

int main() {
  sycl::queue q;
  constexpr int N = 32;
  sycl::nd_range<1> r{sycl::range<1>(N), sycl::range<1>(8)};

  // single_task, non-templated.
  int *p = sycl::malloc_shared<int>(1, q);
  *p = 0;
  syclexp::single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(store42, p));
  q.wait();
  assert(*p == 42);

  // single_task, templated — deduce T = int.
  *p = 0;
  syclexp::single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(store_one, p));
  q.wait();
  assert(*p == 1);

  // nd_range, non-templated.
  float *y = sycl::malloc_shared<float>(N, q);
  for (int i = 0; i < N; ++i)
    y[i] = 2.0f;
  syclexp::nd_launch(q, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(scale, y, 3.0f, N));
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(y[i] == 6.0f);

  // nd_range, templated — deduce T = float.
  float *x = sycl::malloc_shared<float>(N, q);
  const float a = 2.0f;
  for (int i = 0; i < N; ++i) {
    x[i] = static_cast<float>(i);
    y[i] = 1.0f;
  }
  syclexp::nd_launch(q, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(axpy, y, x, a, N));
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(y[i] == a * static_cast<float>(i) + 1.0f);

  // nd_range, templated, from a dependent (function-template) call site:
  // deduction is deferred to instantiation.
  for (int i = 0; i < N; ++i)
    y[i] = 1.0f;
  run_axpy<float>(q, r, y, x, a, N);
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(y[i] == a * static_cast<float>(i) + 1.0f);

  // Zero launch arguments: SYCL_EXT_ONEAPI_KERNEL_FUNCTION(tick) must expand without a dangling
  // comma and launch cleanly.
  syclexp::single_task(q, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(tick));
  q.wait();

  // Mixed template params: Dim=1 given explicitly (non-deducible), T deduced
  // from the pointer/value arguments. Paren-wrap fill_val<1> so the macro does
  // not split on the template-argument comma (harmless here, required in
  // general).
  for (int i = 0; i < N; ++i)
    y[i] = 0.0f;
  syclexp::nd_launch(q, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION((fill_val<1>), y, 7.0f, N));
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(y[i] == 7.0f);

  // Overload set: the builtin selects fill(int*) then fill(float*) purely from
  // the launch argument types.
  int *ip = sycl::malloc_shared<int>(N, q);
  syclexp::nd_launch(q, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill, ip, 5, N));
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(ip[i] == 5);
  syclexp::nd_launch(q, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(fill, y, 8.0f, N));
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(y[i] == 8.0f);

  // The macro is orthogonal to the launch configuration: it also works with the
  // handler-taking overloads (inside a command group) and with the
  // launch_config overload, not just the queue + nd_range form above.

  // single_task(handler&, ...) inside a command group.
  *p = 0;
  syclexp::submit(q, [&](sycl::handler &h) {
    syclexp::single_task(h, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(store42, p));
  });
  q.wait();
  assert(*p == 42);

  // nd_launch(handler&, nd_range, ...) inside a command group.
  for (int i = 0; i < N; ++i)
    y[i] = 2.0f;
  syclexp::submit(q, [&](sycl::handler &h) {
    syclexp::nd_launch(h, r, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(scale, y, 3.0f, N));
  });
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(y[i] == 6.0f);

  // nd_launch(queue, launch_config, ...).
  for (int i = 0; i < N; ++i)
    y[i] = 2.0f;
  syclexp::launch_config<sycl::nd_range<1>> cfg{r};
  syclexp::nd_launch(q, cfg, SYCL_EXT_ONEAPI_KERNEL_FUNCTION(scale, y, 4.0f, N));
  q.wait();
  for (int i = 0; i < N; ++i)
    assert(y[i] == 8.0f);

  sycl::free(p, q);
  sycl::free(y, q);
  sycl::free(x, q);
  sycl::free(ip, q);
  return 0;
}
