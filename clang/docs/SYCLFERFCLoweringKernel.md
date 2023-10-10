Discourse topic details

- Category: [Clang Frontend](https://discourse.llvm.org/c/clang/6)
- Title: "RFC: SYCL kernel lowering"
- Tags: sycl

# SYCL Kernel Lowering

A SYCL construct such as `parallel_for` or `single_task` takes a named function object or a lambda as one of its arguments.   The contents of this function object are executed on the device.   However, as the SYCL runtime can rely on other offload APIs like OpenCL or CUDA to execute the function object, it needs to respect the calling convention of these API.  To enable this, the function object is converted into the format of an OpenCL kernel.

Consider the following code snippet:

```
#include <sycl.hpp>

int main() {
  sycl::queue q;
  sycl::buffer<char> b{sycl::range{1024}};

  q.submit([&](sycl::handler &cgh) {
    sycl::accessor acc{b, cgh};
    int i;
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

In this example, the lambda passed to the `single_task` construct needs to be executed on the device.  The corresponding function object looks like:

```
struct FuncObj {
  int i;
  struct S test_s;
  sycl::accessor acc1;

  void operator () {  // Function call operator
    if (i == 13 && test_s.c == 14) {
      acc[0] = 'a';
    }
  }
};
```

The device compiler then generates a caller in the form of an OpenCL kernel function that calls this function object.  It does so by walking the function object data member and generating a parameter for each of them.  Some special types like accessor are treated a bit differently (see below).  Inside the OpenCL kernel, the function object is rebuilt and then called.

The device compiler transforms this into (pseudo-code):

```
    spir_kernel void Caller(
       int i,
       struct S test_s,
       __global int* accData, // arg1 of accessor init function
       range<1> accR1,        // arg2 of accessor init function
       range<1> accR2,        // arg3 of accessor init function
       id<1> accId            // arg4 of accessor init function
    )
    {
        // Local capture object
        struct FuncObj local;

        // Reassemble capture object from parts
        local.i = i;
        local.s = s;
        // Call acc1 accessor's init function
        sycl::accessor::init(&local.acc1, accData, accR1, accR2, accId);

        // Call the kernel body
        Callee(&local);
    }

    spir_func void Callee(struct FuncObj* this)
    {
        // body of the kernel invocation
    }
```

The SYCL specification defines rules for allowable types for a kernel parameter.

The proposed implementation passes the copyable types to the device as separate parameters.  The current implementation is aware of some types such as sycl::accessor, for example, which cannot be simply copied from host to device.  (The specification permits this to account for difference in host/device layouts, absence of some fields on either the host or the device, or to allow conversion of pointer values for correct behavior.)  To enable all of this, these special types have an __init function.  The parameters of this function are transfered from host to device separately.  The values received on the device are passed to the init functions executed on the device, which results in the reassembly of the SYCL object in a form usable on the device.  Note that when such types are elements of an array or a field of a struct or both, special traversal is necessary to pass the type properly.  The proposed mechanism accounts for handling these special instances.

# Location of this logic

Currently in our implementation, this logic is located in the Sema phase.  Similar to what we are considering for generating the Integration Headers and Foooters, we have an open question between two options that we are considering - one, is to move this to the CodeGen phase and two, move it out of the clang FE and do it in an LLVM IR pass.
