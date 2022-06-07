# SYCL Kernel Parameter Handling and Array Support

## Introduction

This document describes how parameters of SYCL kernels are passed
from host to device. Support for arrays as kernel parameters was added
later and aspects of that design are covered in more detail.
The special treatment of arrays of `sycl::accessor` objects is also discussed.
Array support covers these cases:

1. arrays of standard-layout type
2. arrays of accessors
3. arrays of structs that contain accessor arrays or accessor fields

The motivation for allowing arrays as kernel parameters is to 
bring consistency to the treatment of arrays. 
In C++ a lambda function is allowed to access an element of an array
defined outside the lambda. The compiler captures the entire array
by value. Note that this behavior is limited to implicit 
capture of the array by value. If the array name were in
the capture list then the base address of the array would be captured
and not the entire array.

A user would expect the same mode of array capture in a SYCL kernel 
lambda object as in any other lambda object. 

The first few sections describe the overall design.
The last three sections provide additional details of array support.
The implementation of this design is confined to four classes in the
file `SemaSYCL.cpp`.

## A SYCL Kernel

The SYCL constructs `single_task`, `parallel_for`, and
`parallel_for_work_group` each take a function object or a lambda function
 as one of their arguments. The code within the function object or
lambda function is executed on the device.
To enable execution of the kernel on OpenCL devices, the lambda/function object
is converted into the format of an OpenCL kernel.

## SYCL Kernel Code Generation

Consider a source code example that captures an int, a struct and an accessor
by value:

```C++
constexpr size_t c_num_items = 10;
range<1> num_items{c_num_items}; // range<1>(num_items)

int main()
{
  int output[c_num_items];
  queue myQueue;

  int i = 55;
  struct S {
    int m;
  } s = { 66 };
  auto outBuf = buffer<int, 1>(&output[0], num_items);

  myQueue.submit([&](handler &cgh) {
    auto outAcc = outBuf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class Worker>(num_items, [=](cl::sycl::id<1> index) {
      outAcc[index] = i + s.m;
    });
  });

  return 0;
}
```

The input to the code generation routines is a function object that represents
the kernel. In pseudo-code:

```C++
struct Capture {
    sycl::accessor outAcc;
    int i;
    struct S s;
    () {
        outAcc[index] = i + s.m;
    }
}
```

The compiler-generated code for a call to such a lambda function would look like this:
```C++
()(struct Capture* this);
```
       
When offloading the kernel to a device, the lambda/function object's 
function operator cannot be directly called with a capture object address.
Instead, the code generated for the device is in the form of a
"kernel caller" and a "kernel callee".
The callee is a clone of the SYCL kernel object.
The caller is generated in the form of an OpenCL kernel function.
It receives the lambda capture object in pieces, assembles the pieces
into the original lambda capture object and then calls the callee:

```C++
spir_kernel void caller(
    __global int* AccData, // arg1 of accessor init function
    range<1> AccR1,        // arg2 of accessor init function
    range<1> AccR2,        // arg3 of accessor init function
    id<1> I,               // arg4 of accessor init function
    int i,
    struct S s
)
{
    // Local capture object
    struct Capture local;

    // Reassemble capture object from parts
    local.i = i;
    local.s = s;
    // Call accessor's init function
    sycl::accessor::init(&local.outAcc, AccData, AccR1, AccR2, I);

    // Call the kernel body
    callee(&local, id<1> wi);
}

spir_func void callee(struct Capture* this, id<1> wi)
{
}
```

As may be observed from the example above, standard-layout lambda capture
components are passed by value to the device as separate parameters.
This includes scalars, pointers, and standard-layout structs.
Certain object types defined by the SYCL standard, such as 
`sycl::accessor` and `sycl::sampler` although standard-layout, cannot be 
simply copied from host to device. Their layout on the device may be different
from that on the host. Some host fields may be absent on the device,
other host fields replaced with device-specific fields and
the host data pointer field must be translated to an OpenCL
or L0 memory object before it can be passed as a kernel parameter.
To enable all of this, the parameters of the `sycl::accessor`
and `sycl::sampler` init functions are transfered from 
host to device separately. The values received on the device
are passed to the `init` functions executed on the device,
which results in the reassembly of the SYCL object in a form usable on the device.

There is one other aspect of code generation. An "integration header"
is generated for use during host compilation.
This header file contains entries for each kernel.
Among the items it defines is a table of sizes and offsets of the
kernel parameters.
For the source example above the integration header contains the
following snippet:

```C++
// array representing signatures of all kernels defined in the
// corresponding source
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE19->18clES2_E6Worker
  { kernel_param_kind_t::kind_accessor, 4062, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 32 },
  { kernel_param_kind_t::kind_std_layout, 4, 36 },
};
```

Each entry in the kernel_signatures table is a `kernel_param_desc_t`
object which contains three values:
1) an encoding of the type of capture object member
2) a field that encodes additional properties, and
3) an offset within the lambda object where the value of that kernel argument is placed

The previous sections described how kernel arguments are handled today.
The next three sections describe support for arrays.

## Fix 1: Kernel Arguments that are Standard-Layout Arrays

As described earlier, each variable captured by a lambda that comprises a
SYCL kernel becomes a parameter of the kernel caller function.
For arrays, simply allowing them through would result in a
function parameter of array type. This is not supported in C++.
Therefore, the array needing capture is decomposed into its elements for
the purposes of passing to the device. Each array element is passed as a
separate parameter. The array elements received on the device
are copied into the array within the local capture object.

**Source code fragment:**

```C++
  constexpr int num_items = 2;
  int array[num_items];
  int output[num_items];

  auto outBuf = buffer<int, 1>(&output[0], num_items);

  myQueue.submit([&](handler &cgh) {
    auto outAcc = outBuf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class Worker>(num_items, [=](cl::sycl::id<1> index) {
      outAcc[index] = array[index.get(0)];
    });
  });
```

**Integration header produced:**

```C++
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ1fRN2cl4sycl5queueEENK3$_0clERNS0_7handlerEE6Worker
  { kernel_param_kind_t::kind_accessor, 4062, 0 },
  { kernel_param_kind_t::kind_std_layout, 4, 32 },
  { kernel_param_kind_t::kind_std_layout, 4, 36 },

};

```

**The changes to device code made to support this extension, in pseudo-code:**

```C++
struct Capture {
    sycl::accessor outAcc;
    int array[num_items];
    () {
        // Body
    }
}

spir_kernel void caller(
    __global int* AccData, // arg1 of accessor init function
    range<1> AccR1,        // arg2 of accessor init function
    range<1> AccR2,        // arg3 of accessor init function
    id<1> I,               // arg4 of accessor init function
    int p_array_0;         // Pass array element 0
    int p_array_1;         // Pass array element 1
)
{
    // Local capture object
    struct Capture local;

    // Reassemble capture object from parts
    // Initialize array using existing clang Initialization mechanisms
    local.array[0] = p_array_0;
    local.array[1] = p_array_1; 
    // Call accessor's init function
    sycl::accessor::init(&local.outAcc, AccData, AccR1, AccR2, I);

    callee(&local, id<1> wi);
}
```

## Fix 2: Kernel Arguments that are Arrays of Accessors

Arrays of accessors are supported in a manner similar to that of a plain
accessor. For each accessor array element, the four values required to
call its init function are passed as separate arguments to the kernel.
Reassembly within the kernel caller is done by calling the `init` functions
of each accessor array element in ascending index value.

**Source code fragment:**

```C++
  myQueue.submit([&](handler &cgh) {
    using Accessor =
        accessor<int, 1, access::mode::read, access::target::global_buffer>;
    Accessor inAcc[2] = {in_buffer1.get_access<access::mode::read>(cgh),
                         in_buffer2.get_access<access::mode::read>(cgh)};
    auto outAcc = out_buffer.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class Worker>(num_items, [=](cl::sycl::id<1> index) {
      outAcc[index] = inAcc[0][index] + inAcc[1][index];
    });
  });
```

**Integration header:**

```C++
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE20->18clES2_E6Worker
  { kernel_param_kind_t::kind_accessor, 4062, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 32 },
  { kernel_param_kind_t::kind_accessor, 4062, 64 },
};
```

**Device code generated in pseudo-code form:**

```C++
struct Capture {
    sycl::accessor outAcc; 
    sycl::accessor inAcc[2];
    () {
        // Body
    }
}

spir_kernel void caller(
    __global int* outAccData,    // args of OutAcc
    range<1> outAccR1,
    range<1> outAccR2,
    id<1> outI, 
    __global int* inAccData_0,   // args of inAcc[0]
    range<1> inAccR1_0,
    range<1> inAccR2_0,
    id<1> inI_0,
    __global int* inAccData_1,   // args of inAcc[1]
    range<1> inAccR1_1,
    range<1> inAccR2_1,
    id<1> inI_1,
)
{
    // Local capture object
    struct Capture local;

    // Reassemble capture object from parts 
    // Call outAcc accessor's init function
    sycl::accessor::init(&local.outAcc, outAccData, outAccR1, outAccR2, outI); 

    // Call inAcc[0] accessor's init function
    sycl::accessor::init(&local.inAcc[0], inAccData_0, inAccR1_0, inAccR2_0, inI_0);

    // Call inAcc[1] accessor's init function
    sycl::accessor::init(&local.inAcc[1], inAccData_1, inAccR1_1, inAccR2_1, inI_1);

    callee(&local, id<1> wi);
}
```

## Fix 3: Accessor Arrays within Structs

Kernel parameters that are structs are traversed member
by member, recursively, to enumerate member structs that are one of
the SYCL special types: `sycl::accessor` and `sycl::sampler`.
The arguments of the `init` functions of each special struct encountered 
in the traversal are added as separate arguments to the kernel.
Support for arrays containing SYCL special types
builds upon the support for single accessors within structs.
Each element of such arrays is treated as
an individual object, and the arguments of its init function
are added to the kernel arguments in sequence.
Within the kernel caller function, the lambda object is reassembled
in a manner similar to other instances of accessor arrays. 


**Source code fragment:**

```C++
 myQueue.submit([&](handler &cgh) {
   using Accessor =
        accessor<int, 1, access::mode::read, access::target::global_buffer>;
   struct S {
     int m;
     sycl::accessor inAcc[2];
   } s = { 55,
           {in_buffer1.get_access<access::mode::read>(cgh),
            in_buffer2.get_access<access::mode::read>(cgh)}
   };
   auto outAcc = out_buffer.get_access<access::mode::write>(cgh);

   cgh.parallel_for<class Worker>(num_items, [=](cl::sycl::id<1> index) {
     outAcc[index] = s.m + s.inAcc[0][index] + s.inAcc[1][index];
   });
});
```

**Integration header:**

```C++
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE20->18clES2_E6Worker
  { kernel_param_kind_t::kind_accessor, 4062, 0 },
  { kernel_param_kind_t::kind_std_layout, 72, 32 },
  { kernel_param_kind_t::kind_accessor, 4062, 40 },
  { kernel_param_kind_t::kind_accessor, 4062, 72 },

};
```

**Device code generated in pseudo-code form:**

```C++
struct Capture {
    sycl::accessor outAcc;
    struct S s;
    () {
        // Body
    }
}

spir_kernel void caller(
    __global int* outAccData,  // args of OutAcc
    range<1> outAccR1,
    range<1> outAccR2,
    id<1> outI,
    struct S s,                // the struct S
    __global int* inAccData_0, // args of s.inAcc[0]
    range<1> inAccR1_0,
    range<1> inAccR2_0,
    id<1> inI_0,
    __global int* inAccData_1, // args of s.inAcc[1]
    range<1> inAccR1_1,
    range<1> inAccR2_1,
    id<1> inI_1,
)
{
    // Local capture object
    struct Capture local;

    // Reassemble capture object from parts

    // 1. Copy struct argument contents to local copy
    local.s = s;

    // 2. Initialize accessors by calling init functions
    // 2a. Call outAcc accessor's init function
    sycl::accessor::init(
       &local.outAcc, outAccData, outAccR1, outAccR2, outI); 

    // 2b. Call s.inAcc[0] accessor's init function
    sycl::accessor::init(
       &local.s.inAcc[0], inAccData_0, inAccR1_0, inAccR2_0, inI_0);

    // 2c. Call s.inAcc[1] accessor's init function
    sycl::accessor::init(
       &local.s.inAcc[1], inAccData_1, inAccR1_1, inAccR2_1, inI_1);

    callee(&local, id<1> wi);
}
```
