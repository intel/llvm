<h2>Array Parameters of SYCL Kernels</h2>

<h3>Introduction</h3>

This document describes the changes to support passing arrays to SYCL kernels 
and special treatment of Accessor arrays.
The following cases are handled:

1. arrays of standard-layout type as top-level arguments
2. arrays of Accessors as top-level arguments
3. arrays of accessors within structs that are top-level arguments

The motivation for this correction to kernel parameters processing is to 
bring consistency to the treatment of arrays. 
On the CPU, a lambda function is allowed to access an element of an array
defined outside the lambda. The implementation captures the entire array
by value. A user would naturally expect this to work in SYCL as well. 
However, the current implementation flags references to arrays defined
outside a SYCL kernel as errors.

The first few sections describe the current design.
The last three sections describe the design to support 1. to 3. above.
The implementation of this design is confined to three functions in the
file `SemaSYCL.cpp`.

<h3>A SYCL Kernel</h3>

The SYCL constructs `single_task`, `parallel_for`, and
`parallel_for_work_group` each take a function object or a lambda function
 as one of their arguments. The code within the function object or
lambda function is executed on the device.
To enable execution of the kernel on OpenCL devices, the lambda/function object
is converted into the format of an OpenCL kernel.

<h3>SYCL Kernel Code Generation</h3>

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
    Accessor outAcc;
    int i;
    struct S s;
    () {
        outAcc[index] = i + s.m;
    }
}
```

On the CPU a call to such a lambda function would look like this:
```C++
()(struct Capture* this);
```
       
When offloading the kernel to a device, the lambda/function object's 
function operator cannot be directly called with a capture object address.
Instead, the code generated for the device is in the form of a
“kernel caller” and a “kernel callee”.
The callee is a clone of the SYCL kernel object.
The caller is generated in the form of an OpenCL kernel function.
It receives the lambda capture object in pieces, assembles the pieces
into the original lambda capture object and then calls the callee:

```C++
spir_kernel void caller(
    __global int* AccData, // arg1 of Accessor init function
    range<1> AccR1,        // arg2 of Accessor init function
    range<1> AccR2,        // arg3 of Accessor init function
    id<1> I,               // arg4 of Accessor init function
    int i,
    struct S s
)
{
    // Local capture object
    struct Capture local;

    // Reassemble capture object from parts
    local.i = i;
    local.s = s;
    // Call accessor’s init function
    Accessor::init(&local.outAcc, AccData, AccR1, AccR2, I);

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
Certain SYCL struct types that are not standard-layout,
such as Accessors and Samplers, are treated specially.
The arguments to their init functions are passed as separate parameters
and used within the kernel caller function to initialize Accessors/Samplers
on the device by calling their init functions using the received arguments.

There is one other aspect of code generation. An “integration header”
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

Each entry in the kernel_signatures table contains three values:
1) an encoding of the type of capture object member
2) a field that encodes additional properties, and
3) an offset within a block of memory where the value of that
4) kernel argument is placed.

The previous sections described how kernel arguments are handled today.
The next three sections describe support for arrays.

<h3>Fix 1: Kernel Arguments that are Standard-Layout Arrays</h3>

As described earlier, each variable captured by a lambda that comprises a
SYCL kernel becomes a parameter of the kernel caller function.
For arrays, simply allowing them through would result in a
function parameter of array type. This is not supported in C++.
Therefore, the array needing capture is wrapped in a struct for
the purposes of passing to the device. Once received on the device
within its wrapper, the array is copied into the local capture object.
All references to the array within the kernel body are directed to
the non-wrapped array which is a member of the local capture object.

<h4>Source code fragment:</h4>

```C++
  int array[100];
  auto outBuf = buffer<int, 1>(&output[0], num_items);

  myQueue.submit([&](handler &cgh) {
    auto outAcc = outBuf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class Worker>(num_items, [=](cl::sycl::id<1> index) {
      outAcc[index] = array[index.get(0)];
    });
  });
```

<h4>Integration header produced:</h4>

```C++
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE16->18clES2_E6Worker
  { kernel_param_kind_t::kind_accessor, 4062, 0 },
  { kernel_param_kind_t::kind_std_layout, 400, 32 },
};
```

<h4>The changes to device code made to support this extension, in pseudo-code:</h4>

```C++
struct Capture {
    Accessor outAcc;
    int array[100];
    () {
        // Body
    }
}

struct wrapper {
    int array[100];
};
spir_kernel void caller(
    __global int* AccData, // arg1 of Accessor init function
    range<1> AccR1,        // arg2 of Accessor init function
    range<1> AccR2,        // arg3 of Accessor init function
    id<1> I,               // arg4 of Accessor init function
    struct wrapper w_s     // Pass the array wrapped in a struct
)
{
    // Local capture object
    struct Capture local;

    // Reassemble capture object from parts
    // Initialize array using existing clang Initialization mechanisms
    local.array = w_s; 
    // Call accessor’s init function
    Accessor::init(&local.outAcc, AccData, AccR1, AccR2, I);

    callee(&local, id<1> wi);
}
```

The sharp-eyed reviewer of `SemaSYCL.cpp` will notice that the array
is actually double-wrapped in structs. This was done simply to preserve
the interface to an existing function (`CreateAndAddPrmDsc`) which
processes each kernel caller parameter as a capture object member. 
The object being added to a list in `CreateAndAddPrmDsc` is `Fld`,
which is expected to be a field of some struct. So a wrapped struct
cannot be passed to this function. A double-wrapped struct is needed
as shown below. This does not affect the generated code.

```C++
struct {
  struct {
    int array[100];
  }
}
```

This could be changed but it would mean changes to the `CreateAndAddPrmDsc`
implementation, to all its callers and to the place where the list created
by it is processed.
By wrapping the array twice, the inner, single-wrapped array appears as a
member of a struct and meets the requirements of the existing code.

<h3>Fix 2: Kernel Arguments that are Arrays of Accessors</h3>

Arrays of accessors are supported in a manner similar to that of a plain
Accessor. For each accessor array element, the four values required to
call its init function are passed as separate arguments to the kernel.
Reassembly within the kernel caller is serialized by accessor array element.

<h4>Source code fragment:</h4>

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

<h4>Integration header:</h4>

```C++
static constexpr
const kernel_param_desc_t kernel_signatures[] = {
  //--- _ZTSZZ4mainENKUlRN2cl4sycl7handlerEE20->18clES2_E6Worker
  { kernel_param_kind_t::kind_accessor, 4062, 0 },
  { kernel_param_kind_t::kind_accessor, 4062, 32 },
  { kernel_param_kind_t::kind_accessor, 4062, 64 },
};
```

<h4>Device code generated in pseudo-code form:</h4>

```C++
struct Capture {
    Accessor outAcc; 
    Accessor inAcc[2];
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
    // Call outAcc accessor’s init function
    Accessor::init(&local.outAcc, outAccData, outAccR1, outAccR2, outI); 

    // Call inAcc[0] accessor’s init function
    Accessor::init(&local.inAcc[0], inAccData_0, inAccR1_0, inAccR2_0, inI_0);

    // Call inAcc[1] accessor’s init function
    Accessor::init(&local.inAcc[1], inAccData_1, inAccR1_1, inAccR2_1, inI_1);

    callee(&local, id<1> wi);
}
```

<h3>Fix 3: Accessor Arrays within Structs</h3>

*Individual* Accessors within structs were already supported.
Struct parameters of kernels that are structs are traversed member
by member, recursively, to enumerate member structs that are one of
the SYCL special types: Accessors and Samplers. For each special
struct encountered in the scan, arguments of their init functions
are added as separate arguments to the kernel.
However, *arrays* of accessors within structs were not supported.
Building on the support for single Accessors within structs,
the extension to arrays of Accessors/Samplers within structs
is straightforward. Each element of such arrays is treated as
an individual object, and the arguments of its init function
are added to the kernel arguments in sequence.
Within the kernel caller function, the lambda object is reassembled
in a manner similar to other instances of Accessor arrays. 


<h4>Source code fragment:</h4>

```C++
 myQueue.submit([&](handler &cgh) {
   using Accessor =
        accessor<int, 1, access::mode::read, access::target::global_buffer>;
   struct S {
     int m;
     Accessor inAcc[2];
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

<h4>Integration header:</h4>

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

<h4>Device code generated in pseudo-code form:</h4>

```C++
struct Capture {
    Accessor outAcc;
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
    // Copy struct argument contents to local copy
    // Accessor array will be initialized by calling init functions
    local.s = s;

    // Call outAcc accessor’s init function
    Accessor::init(
       &local.outAcc, outAccData, outAccR1, outAccR2, outI); 

    // Call s.inAcc[0] accessor’s init function
    Accessor::init(
       &local.s.inAcc[0], inAccData_0, inAccR1_0, inAccR2_0, inI_0);

    // Call s.inAcc[1] accessor’s init function
    Accessor::init(
       &local.s.inAcc[1], inAccData_1, inAccR1_1, inAccR2_1, inI_1);

    callee(&local, id<1> wi);
}
```
