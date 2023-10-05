# SYCL Kernel

A SYCL construct such as `parallel_for` or 'single_task' takes a a named function object or a lambda as one of its arguments.   The contents of this function object are executed on the device.   However, as the SYCL runtime can rely on other offload APIs like OpenCL or CUDA to execute the function object, it needs to respect the calling convention of these API.  To enable this, the function object is converted into the format of an OpenCL kernel.

Consider the following code snippet:

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

In this example, the lambda passed to the 'kernel_single_task' construct needs to be executed on the device.  To do this, we propose generating a function object representing the kernel.   The function call operator of this object has the contents of the kernel invocation.  The function object generated looks like:

```
struct FuncObj {
  int i;
  struct S test_s;
  sycl::accessor acc1;
  sycl::sampler smplr;

  void operator () {  // Function call operator
    if (i == 13 && test_s.c == 14) {
      acc1.use();
      smplr.use();
    }
  }
};
```

The device compiler then generates a caller in the form of an OpenCL kernel function that calls this function object.  It does so by walking the function object data member and producing a parameter for each of them.  Inside the OpenCL kernel, the function object is rebuilt and then called.  Some special types like accessor and sampler are treated a bit differently (see below).

The device compiler transforms this into (pseudo-code):

```
    spir_kernel void Caller(
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
        Callee(&local);
    }

    spir_func void Callee(struct FuncObj* this)
    {
        // body of the kernel invocation
    }
```

The SYCL specification defines rules for allowable types for a kernel parameter.

The proposed implementation passes the copyable types to the device as separate parameters.  The current implementation is aware of some types such as sycl::accessor and sycl::sampler, for example, which cannot be simply copied from host to device.  (The specification permits this to account for difference in host/device layouts, absence of some fields on either the host or the device, or to allow conversion of pointer values for correct behavior.)  To enable all of this, the parameters of the sycl::accessor and sycl::sampler init functions are transfered from host to device separately.  The values received on the device are passed to the init functions executed on the device, which results in the reassembly of the SYCL object in a form usable on the device.  Note that when such types are elements of an array or a field of a struct or both, special traversal is necessary to pass the type properly.  The proposed mechanism accounts for handling these special instances.
