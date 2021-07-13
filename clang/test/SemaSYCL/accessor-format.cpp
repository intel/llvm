// RUN: %clang_cc1 -fsycl-is-device -verify -DVALID_ACCESSOR %s
// RUN: %clang_cc1 -fsycl-is-device -verify -DTOO_FEW_PARAM %s
// RUN: %clang_cc1 -fsycl-is-device -verify -DINV_PARAM_TYPE %s

// Test which verifies that incorrect accessor format, in rogue
// libraries, does not cause the compiler to crash.

namespace cl {
namespace sycl {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class placeholder { false_t,
                         true_t };
} // namespace access

#ifdef VALID_ACCESSOR
template <typename DataT, int dimensions = 1,
          access::mode AccessMode = access::mode::read_write,
          access::target AccessTarget = access::target::global_buffer,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
#elif TOO_FEW_PARAM
template <typename DataT,
          access::mode AccessMode = access::mode::read_write,
          access::target AccessTarget = access::target::global_buffer,
          access::placeholder IsPlaceholder = access::placeholder::false_t>
#elif INV_PARAM_TYPE
template <typename DataT, typename dimensions = int,
          typename InvalidAccessModeType = int,
          int InvalidTargetType = 0,
          char InvalidTargetTypePlaceHolderType = 'T'>
#endif
class accessor {
public:
  void use() const {}

private:
  void __init() {}
};

} // namespace sycl
} // namespace cl

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {

#ifdef VALID_ACCESSOR
  cl::sycl::accessor<int> AccessorValid;
  cl::sycl::accessor<int, 1, 1024> AccessorValidLibInvalidUserCode; // expected-error{{value of type 'int' is not implicitly convertible to 'access::mode'}}
  kernel_single_task<class fake_kernel>([=]() {
    AccessorValid.use();
  });
#elif TOO_FEW_PARAM
  // expected-error@#TooFewParam{{accessor class template must have atleast five template parameters}}
  cl::sycl::accessor<int> AccessorInvalid;
  kernel_single_task<class fake_kernel>([=]() {
    AccessorInvalid.use(); // #TooFewParam
  });
#elif INV_PARAM_TYPE
  // expected-error@#InvalidTypes{{second template parameter of the accessor class template must be of integer type}}
  // expected-error@#InvalidTypes{{third template parameter of the accessor class template must be of access::mode enum type}}
  // expected-error@#InvalidTypes{{fourth template parameter of the accessor class template must be of access::target enum type}}
  // expected-error@#InvalidTypes{{fifth template parameter of the accessor class template must be of access::placeholder enum type}}
  cl::sycl::accessor<int> AccessorInvalid;
  kernel_single_task<class fake_kernel>([=]() {
    AccessorInvalid.use(); // #InvalidTypes
  });
#endif

  return 0;
}
