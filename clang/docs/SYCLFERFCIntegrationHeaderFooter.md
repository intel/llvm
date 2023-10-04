# Integration Header

We propose an Integration Header for two purposes:

* to allow a non-SYCL-aware compiler to do host compilations and
* as a mechanism permitting the SYCL run-time library to use the information provided for its needs

Consider the following partial code:

```
  sycl::accessor<char, 1, sycl::access::mode::read> acc1;
  int i = 13;
  sycl::sampler smplr;
  struct S {
    char c;
    int i;
  } test_s;
  test_s.c = 14;
  kernel_single_task<class first_kernel>([=]() {
    if (i == 13 && test_s.c == 14) {

      acc1.use();
      smplr.use();
    }
  });
```

Here the lambda within the 'kernel_single_task' construct needs to be executed on the device.  To do this, a function object representing the kernel is generated.   The function call operator of this object has the contents of the kernel invocation.  The function object generated looks like:

```
struct FuncObj {
  int i;
  struct S test_s;
  sycl::accessor acc1;
  sycl::sampler smplr;

  () {  // Function call operator
    if (i == 13 && test_s.c == 14) {
      acc1.use();
      smplr.use();
    }
  }
};
```

The device compiler then generates a caller in the form of an OpenCL kernel function that calls this function object.

The device compiler transforms this into (pseudo-code):

```
    spir_kernel void caller(
       int i,
       struct S test_s,
       __global int* accData, // arg1 of accessor init function
       range<1> accR1,        // arg2 of accessor init function
       range<1> accR2,        // arg3 of accessor init function
       id<1> accId,           // arg4 of accessor init function
       sampler_t smpData      // arg1 of sampler init function
    )
    {
        // Local capture object
        struct FuncObj local;

        // Reassemble capture object from parts
        local.i = i;
        local.s = s;
        // Call acc1 accessor's init function
        sycl::accessor::init(&local.acc1, accData, accR1, accR2, accId);
        // Call smplr sampler's init function
        sycl::accessor::init(&local.smplr, smpData);

        // Call the kernel body
        callee(&local);
    }

    spir_func void callee(struct FuncObj* this)
    {
        // body of the kernel invocation
    }
```

For details of this transformation, see 'Lowering of SYCL-Kernel'. // TODO: Add link

The integration header then describes the fields of the kernel objects as follows:

``
namespace sycl {
  inline namespace _V1 {
    namespace detail {
      // names of all kernels defined in the corresponding source
      static constexpr
      const char* const kernel_names[] = {
        "_ZTSZ4mainE12first_kernel"  // mangled name associated with first_kernel
                                     // demangles to 'typeinfo name for main::first_kernel'

      };
           // array representing signatures of all kernels defined in the
           // corresponding source
          static constexpr
          const kernel_param_desc_t kernel_signatures[] = {
              //--- _ZTSZ4mainE12first_kernel
             { kernel_param_kind_t::kind_std_layout, 4, 0 },
            { kernel_param_kind_t::kind_std_layout, 8, 4 },
            { kernel_param_kind_t::kind_accessor, 4062, 12 },
            { kernel_param_kind_t::kind_sampler, 8, 24 },

           { kernel_param_kind_t::kind_invalid, -987654321, -987654321 },
         };

     }

  }

}
```

For each kernel, a mangled name associated with it, is stored in the kernel_names array.

And for each kernel, each of its parameter is described in the kernel_signatures array.   The description consists of the following parts:

* an encoding of the type that is captured
* its size and its offset

In the above example, the first two parameters are standard layout types (the int 'i' and the struct 'test_s'), the latter two though standard layout types are types which need special handling - accessor and sampler respectively.   The last entry acts as a terminating entry.

This allows the run-time library to recreate the function object on the device.

The full set of encoding is:

* kind_accessor  for sycl accessor types
* kind_std_layout for standard layout types
* kind_sampler for sycl sampler types
* kind_pointer for pointer types
* kind_specialization_constants_buffer for initializing specialization constants
* kind_stream for sycl stream types and
* kind_invalid used to indicate the terminal value in the signatures array.


# Integration Footer


We propose an Integration Footer primarily to support SYCL specialization constants.

The SYCL 2020 Specification defines a specialization constant as a constant variable where the value is not known until compilation of the SYCL kernel function.  In order to accommodate both implementations that have native support for this as well as those that do not, we propose the following mechanism:

The header files declare, but not define, a special function to obtain the numeric id of a specialization constant.
```
     namespace detail {
         template<auto &SpecConstName>
         inline const char *get_spec_constant_symbolic_ID();
     }
```

Definition of that function template is provided by DPC++ FE in form of an integration footer file:
```
     namespace detail {
       // assuming user defined the following specialization_id:
       // constexpr specialiation_id<int> int_const;
       // class Wrapper {
       // public:
       //   static constexpr specialization_id<float> float_const;
       // };

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

This footer file is appended at the end of each translation unit for the host compilation.


All specialization constants used within a program are bundled together and stored into a single buffer, which is passed as implicit kernel argument. The layout of that buffer is well-defined and known to both the compiler and the runtime, so when user sets the value of a specialization constant, that value is being copied into particular place within that buffer and once the constant is requested in device code, the compiler generates a load from the same place of the buffer.

# Usage of Integration Headers and Footers

Integration headers and footers are generated by the device compiler only if necessary.   Headers are necessary if there is a sycl kernel invocation and footers are necessary if specialization constants are used.  They are then passed to the host compiler.   The integration header is included before the TU is processed by the host compiler and the integration footer is appended to the end of the TU.
