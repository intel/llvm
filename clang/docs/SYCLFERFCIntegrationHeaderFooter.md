# Introduction
SYCL source files are logically (and usually physically) compiled multiple times; code that runs on the "host" (e.g., a CPU) is compiled by a host compiler and code that runs on a "device" (e.g., a GPU) is compiled by a device compiler. The SYCL language is designed such that it is not necessary that the host compiler be SYCL-aware; an ISO C++ conforming compiler suffices. However, some SYCL features such as kernel invocation and support for
[specialization constants](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#_specialization_constants)
require coordination between the host and device compilers. Integration headers and footers produced by a device compiler are used to communicate the information needed during host compilation to support these features.

# Integration Header
An integration header is a source file produced during device compilation that is intended to be pre-included during host compilation of the same source file.

Consider the following code:

```
#include <sycl.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<char> b{sycl::range{1024}};

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{b, cgh};
    int i = 13;
    struct S {
      char c;
      int i;
    } test_s;
    test_s.c = 14;

    cgh.single_task([=] {
      if (i == 13 && test_s.c == 14) {
        acc[0] = 'a';
      }
    });
  });
}
```

Here the lambda passed to the `kernel_single_task` construct needs to be executed on the device.  The corresponding function object looks like:

```
struct FuncObj {
  int i;
  struct S test_s;
  sycl::accessor acc;

  operator void () {  // Function call operator
    if (i == 13 && test_s.c == 14) {
      acc[0] = 'a';
    }
  }
};
```

The function call operator of this object has the contents of the kernel invocation.  The device compiler then generates a caller in the form of an OpenCL kernel function that calls this function object.

For details of this transformation, see `Lowering of SYCL-Kernel`. // TODO: Add link

The integration header then describes the fields of the kernel objects as follows:

```
namespace sycl {
  inline namespace _V1 {
    namespace detail {
      // names of all kernels defined in the corresponding source
      static constexpr
      const char* const kernel_names[] = {
        "unique_name_for_first_kernel"      // first_kernel
      };
      // array representing signatures of all kernels defined in the
      // corresponding source
      static constexpr
      const kernel_param_desc_t kernel_signatures[] = {
        //--- first_kernel
        { kernel_param_kind_t::kind_std_layout, 4, 0 },
        { kernel_param_kind_t::kind_std_layout, 8, 4 },
        { kernel_param_kind_t::kind_accessor, 4062, 12 },

        { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
      };
    } // namespace detail
  } // namespace _V1
} // namespace sycl
```

For each kernel, a mangled name associated with it, is stored in the kernel_names array.

And for each kernel, each of its parameter is described in the kernel_signatures array.   The description consists of the following parts:

* an encoding of the type that is captured
* its size and its offset

In the above example, the first two parameters are standard layout types (the int `i` and the struct `test_s`), the third one, although a standard layout type, is a type which needs special handling.   The last entry acts as a terminating entry.

This allows the run-time library to recreate the function object on the device.

Some of the encoding that we use are:

* kind_accessor  for sycl accessor types
* kind_std_layout for standard layout types
* kind_pointer for pointer types
* kind_specialization_constants_buffer for initializing specialization constants
* kind_stream for sycl stream types and
* kind_invalid used to indicate the terminal value in the signatures array.


# Integration Footer
An integration footer is a source file produced during device compilation that is intended to be post-included during host compilation of the same source file.

The SYCL 2020 Specification defines a specialization constant as a constant variable where the value is not known until compilation of the SYCL kernel function.

Consider the following example:
```
  constexpr specialiation_id<int> int_const;
  class Wrapper {
  public:
    static constexpr specialization_id<float> float_const;
  };
```

Also note that the SYCL header files declare, but not define, a special function to obtain the numeric id of a specialization constant.
```
     namespace detail {
         template<auto &SpecConstName>
         inline const char *get_spec_constant_symbolic_ID();
     }
```


Partial specializations of this function are defined in the integration footer file:
```
   namespace detail {
     template<>
     inline const char *get_spec_constant_symbolic_ID<int_const>() {
       return "unique_name_for_int_const";
     }
     template<>
     inline const char *get_spec_constant_symbolic_ID<Wrapper::float_const>() {
       return "unique_name_for_Wrapper_float_const";
     }
   }
```

This footer file is appended at the end of the translation unit for the host compilation.
