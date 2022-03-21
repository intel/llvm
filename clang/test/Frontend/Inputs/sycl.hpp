#pragma once

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))

inline namespace cl {
namespace sycl {

// Dummy aspect enum with limited enumerators
enum class aspect {
  host = 0,
  cpu = 1,
  gpu = 2,
  accelerator = 3,
  custom = 4,
  fp16 = 5,
  fp64 = 6,
};

class kernel {};
class context {};
class device {};
class event {};

class queue {
public:
  template <typename T>
  event submit(T cgf) { return event{}; }

  void wait() {}
  void wait_and_throw() {}
  void throw_asynchronous() {}
};

class auto_name {};
template <typename Name, typename Type>
struct get_kernel_name_t {
  using name = Name;
};
template <typename Type>
struct get_kernel_name_t<auto_name, Type> {
  using name = Type;
};

class kernel_handler {
  void __init_specialization_constants_buffer(char *specialization_constants_buffer) {}
};

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))
template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_single_task(const KernelType &kernelFunc) { // #KernelSingleTask
  kernelFunc();
}

class handler {
public:
  template <typename KernelName = auto_name, typename KernelType>
  void single_task(const KernelType &kernelFunc) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<NameT>(kernelFunc);
#else
    kernelFunc();
#endif
  }
};

} // namespace sycl
} // namespace cl
