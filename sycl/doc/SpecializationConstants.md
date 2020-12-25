# Specialization constants

DPC++ implements this [proposal](https://github.com/codeplaysoftware/standards-proposals/blob/master/spec-constant/index.md)
with some restrictions. See this [document](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/SpecConstants/README.md) for more details.

## Requirements:

- must work with separate compilation and linking
- must support AOT compilation

Implementation is based on SPIR-V specialization constants. But there is one
important difference between SYCL and SPIR-V: in SYCL specialization constants
are identified by a type ID which is mapped to a symbolic name, in SPIR-V - by
an ordinal number. This complicates the design, as the compiler
1) needs to propagate symbolic =\> numeric ID correspondence to the runtime
2) can assign numeric IDs only when linking due to the separate compilation

Simple source code example:

```
struct A {
  int x;
  float y;
};

struct POD {
  A a[2];
  // FIXME: cl::sycl::vec class is not a POD type in our implementation by some
  // reason, but there are no limitations for vector types from spec constatns
  // design point of view.
  cl::sycl::vec<int, 2> b;
};

class MyInt32Const;
class MyPODConst;
// ...
  POD gold{
    {
      { goldi, goldf },
      { goldi + 1, goldf + 1 },
    },
    { goldi, goldi }
  };

  sycl::program p(q.get_context());
  sycl::ONEAPI::experimental::spec_constant<int32_t, MyInt32Const> i32 =
      p.set_spec_constant<MyInt32Const>(rt_val);

  cl::sycl::ONEAPI::experimental::spec_constant<POD, MyPODConst> pod =
      p.set_spec_constant<MyPODConst>(gold);

  p.build_with_kernel_type<MyKernel>();
  sycl::buffer<int, 1> buf(vec.data(), vec.size());
  sycl::buffer<POD, 1> buf(vec_pod.data(), vec_pod.size());

  q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
    auto acc_pod = buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task<MyKernel>(
        p.get_kernel<MyKernel>(),
        [=]() {
          acc[0] = i32.get();
          acc_pod[0] = pod.get();
        });
  });
// ...
```

## Design

This document describes both arithmetic and POD types support, because their
handling is mostly unified except a few details.

### DPC++ Headers

`spec_constant` class, which represents a specialization constant in DPC++ also
performs the following tasks:
- provides a mapping from C++ typename, which is used as a specialization
  constant name in DPC++ source file, to a symbolic ID of a specialization
  constant, which is used in the compiler.
- provides a special markup, which allows the compiler to detect
  specialization constants in the device code and properly handle them.

Both tasks are performed in the implementation of the key `spec_constant::get()`
method or the device code:

```
template <typename T, typename ID = T> class spec_constant {
// ...
public:
  // enable_if T is a scalar arithmetic type
  T get() const { // explicit access.
#ifdef __SYCL_DEVICE_ONLY__
    const char *TName = __builtin_unique_stable_name(ID);
    return __sycl_getScalarSpecConstantValue<T>(TName);
#else
    return Val;
#endif // __SYCL_DEVICE_ONLY__

  // enable_if T is a POD type
  T get() const { // explicit access.
#ifdef __SYCL_DEVICE_ONLY__
    const char *TName = __builtin_unique_stable_name(ID);
    return __sycl_getCompositeSpecConstantValue<T>(TName);
#else
    return Val;
#endif // __SYCL_DEVICE_ONLY__
  }
```

here [`__builtin_unique_stable_name`](https://github.com/intel/llvm/blob/sycl/clang/docs/LanguageExtensions.rst#__builtin_unique_stable_name)
is a compiler built-in used to translate types to unique strings, which are
used as symbolic IDs of specialization constants.
`__sycl_getScalarSpecConstantValue<T>` and `__sycl_getCompositeSpecConstant<T>`
are "intrinsics" recognized by a special LLVM pass later.

Compilation and subsequent linkage of the device code results in a number of
`__sycl_getScalarSpecConstantValue` and `__sycl_getCompositeSpecConstantValue`
calls whose arguments are symbolic spec constant IDs. Before generating a device
binary, each linked device code LLVM IR module undergoes processing by
`sycl-post-link` tool which can run LLVM IR passes before passing the module
onto the SPIR-V translator.

### DPC++ Compiler

#### sycl-post-link tool

As it is stated above, the only place where we can properly handle
specialization constants is somewhere during or after linking device code from
different translation units, so it happens in `sycl-post-link` tool.

There is a `SpecConstants` LLVM IR pass which:
1. assigns numeric IDs to specialization constants found in the linked module.
2. brings IR to the form expected by the SPIR-V translator.
3. collects and provides \<Symbolic ID\> =\> \<numeric IDs + additional info\>
   mapping, which is later being used by DPC++ RT to set specialization constant
   values provided by user.

##### Assignment of numeric IDs to specialization constants

This task is achieved by maintaining a map, which for each encountered symbolic
ID of a specialization constant holds a list of numeric IDs, which are used to
identify the specialization constant at SPIR-V level.

NOTE: one symbolic ID can several numeric IDs assigned to it - such 1:N mapping
comes from the fact that at SPIR-V level, composite specialization constants
don't have dedicated IDs and they are being identified and specialized through
their scalar leafs and their numeric IDs.

For example, the following code:
```
struct Nested {
  float a, b;
};
struct A {
  int x;
  Nested n;
};
// ...
sycl::ONEAPI::experimental::spec_constant<int32_t, MyInt32Const> i32 =
    p.set_spec_constant<MyInt32Const>(rt_val);

cl::sycl::ONEAPI::experimental::spec_constant<A, MyPODConst> pod =
    p.set_spec_constant<MyPODConst>(gold);
// ...
  i32.get();
  pod.get();
// ...
```

Will result in the following numeric IDs assignment:
```
// since MyInt32Const is a simple arithmetic specialization constant, we only
// have a single numeric ID associated with its symbolic ID
unique_symbolic_id_for_MyInt32Const -> { 0 }
// MyPODConstant is a composite with three leafs (including nested composite
// types), which results in three numeric IDs associated with the same symbolic
// ID
unique_symbolic_id_for_MyPODConst -> { 1, 2, 3 }
```

As it is shown in the example above, if a composite specialization constant
contains another composite within it, that nested composite is also being
"flattened" and its leafs are considered to be leafs of the parent
specialization constants. This done by depth-first search through the composite
elements.

##### Transformation of LLVM IR to SPIR-V friendly IR form

SPIR-V friendly IR form is a special representation of LLVM IR, where some
function are named in particular way in order to be recognizable by the SPIR-V
translator to convert them into corresponding SPIR-V instructions later.
The format is documented [here](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst).

For specialization constant, we need to generate the following constructs:
```
template<typename T> // T is arithmetic type
T __spirv_SpecConstant(int numericID, T default_value);

template<typename T, typename... Elements> // T is composite type,
// Elements are arithmetic or composite types
T __spirv_SpecConstantComposite(Elements... elements);
```

Particularly, `SpecConstants` pass translates calls to the
`T __sycl_getScalarSpecConstantValue(const char *symbolic_id)` intrinsic into
calls to `T __spirv_SpecConstant(int ID, T default_val)`.
And for `T __sycl_getCompositeSpecConstantValue(const chat *symbolic_id)` it
generates number of `T __spirv_SpecConstant(int ID, T default_val)` calls for
each leaf of the composite type, plus number of
`T __spirv_SpecConstantComposite(Elements... elements)` for each composite type
(including the outermost one).

Example of LLVM IR transformation can be found below, input LLVM IR:
```
%struct.POD = type { [2 x %struct.A], <2 x i32> }
%struct.A = type { i32, float }

%gold_scalar = call i32 __sycl_getScalarSpecConstantValue<POD type mangling> ("MyInt32Const_mangled")
%gold = call %struct.POD __sycl_getCompositeSpecConstantValue<POD type mangling> ("MyPODConst_mangled")
```

LLVM IR generated by `SpecConstants` pass:
```
%gold_scalar = call i32 __spirv_SpecConstant(i32 0, i32 0)

%gold_POD_A0_x = call i32 __spirv_SpecConstant(i32 1, i32 0)
%gold_POD_A0_y = call float __spirv_SpecConstant(i32 2, float 0)

%gold_POD_A0 = call %struct.A __spirv_SpecConstantComposite(i32 %gold_POD_A0_x, float %gold_POD_A0_y)

%gold_POD_A1_x = call i32 __spirv_SpecConstant(i32 3, i32 0)
%gold_POD_A1_y = call float __spirv_SpecConstant(i32 4, float 0)

%gold_POD_A1 = call %struct.A __spirv_SpecConstantComposite(i32 %gold_POD_A1_x, float %gold_POD_A1_y)

%gold_POD_A = call [2 x %struct.A] __spirv_SpecConstantComposite(%struct.A %gold_POD_A0, %struct.A %gold_POD_A1)

%gold_POD_b0 = call i32 __spirv_SpecConstant(i32 4, i32 0)
%gold_POD_b1 = call i32 __spirv_SpecConstant(i32 6, i32 0)
%gold_POD_b = call <2 x i32> __spirv_SpecConstant(i32 %gold_POD_b0, i32 %gold_POD_b1)

%gold = call %struct.POD __spirv_SpecConstantComposite([2 x %struct.A] %gold_POD_A, <2 x i32> %gold_POD_b)

```

###### Ahead of time compilation

With AOT everything is simplified - the `SpecConstants` pass simply replaces
the `__sycl_getScalarSpecConstantValue` calls with constants - default values of
the spec constant's type. No maps are generated, and DPC++ program can't change
the value of a spec constant.

##### Collecting spec constants info and communicating it to DPC++ RT

For each encountered specialization constants `sycl-post-link` emits a property,
which encodes information required by DPC++ RT to set the value of a
specialization constant through corresponding API.

This information is communicated through "SYCL/specialization constants"
property set:
```
// Device binary image property.
// If the type size of the property value is fixed and is no greater than
// 64 bits, then ValAddr is 0 and the value is stored in the ValSize field.
// Example - PI_PROPERTY_TYPE_UINT32, which is 32-bit
struct _pi_device_binary_property_struct {
  char *Name;       // null-terminated property name
  void *ValAddr;    // address of property value
  uint32_t Type;    // _pi_property_type
  uint64_t ValSize; // size of property value in bytes
};
// Named array of properties.
struct _pi_device_binary_property_set_struct {
  char *Name;                                // the name
  pi_device_binary_property PropertiesBegin; // array start
  pi_device_binary_property PropertiesEnd;   // array end
};
struct pi_device_binary_struct {
...
  // Array of property sets; e.g. specialization constants symbol-int ID map is
  // propagated to runtime with this mechanism.
  pi_device_binary_property_set PropertySetsBegin;
  pi_device_binary_property_set PropertySetsEnd;
};
```

So, within a single set we have a separate property for each specialization
constant with name corresponding to its symbolic ID.

Each such property contains an array of tuples (descriptors)
\<leaf spec ID, offset, size\>. This descriptor might be overcomplicated for
simple arithmetic spec constants, but it is still used for them in order to
unify internal representation of scalar and composite spec constants and
simplify their handling in DPC++ RT.
This descriptor is needed, because at DPC++ RT level, composite constants are
set by user as a byte array and we have to break it down to the leaf members of
the composite and set a value for each leaf as for a separate scalar
specialization constant.

For simple scalar specialization constants the array will only contain a single
descriptor representing the constant itself. For composite specialization
constants the array will contain several descriptors for each leaf of the
composite type.

The descriptor contains the following fields:
- ID of a composite constant leaf, i.e. ID of a scalar specialization constant,
  which is a part of a composite type or ID of a constant itself if it is a
  scalar.
- Offset from the beginning of composite, which points to the location of a
  scalar value within the composite, i.e. the position where scalar
  specialization constant resides within the byte array supplied by the user.
  For scalar specialization constants it will always be 0.
- Size of the scalar specialization constant

For example, the following code:
```
struct Nested {
  float a, b;
};
struct A {
  int x;
  Nested n;
};
// ...
sycl::ONEAPI::experimental::spec_constant<int32_t, MyInt32Const> i32 =
    p.set_spec_constant<MyInt32Const>(rt_val);

cl::sycl::ONEAPI::experimental::spec_constant<A, MyPODConst> pod =
    p.set_spec_constant<MyPODConst>(gold);
// ...
  i32.get();
  pod.get();
// ...
```

Will result in the following property set generated:
```
property_set {
  Name = "SYCL/specialization constants",
  properties: [
    property {
      Name: "MyInt32Const_symbolic_ID",
      ValAddr: points to byte array [{0, 0, 4}],
      Type: PI_PROPERTY_TYPE_BYTE_ARRAY,
      Size: sizeof(byte array above)
    },
    property {
      Name: "MyPODConst_symbolic_ID",
      ValAddr: points to byte array [{1, 0, 4}, {2, 4, 4}, {3, 8, 4}],
      Type: PI_PROPERTY_TYPE_BYTE_ARRAY,
      Size: sizeof(byte array above)
    },
  ]
}
```

#### SPIRV-LLVM-Translator

Given the `__spirv_SpecConstant` intrinsic calls produced by the
`SpecConstants` pass:
```
; Function Attrs: alwaysinline
define dso_local spir_func i32 @get() local_unnamed_addr #0 {
  ; args are "ID" and "default value":
  %1 = tail call spir_func i32 @_Z20__spirv_SpecConstantii(i32 42, i32 0)
  ret i32 %1
}

%struct.A = type { i32, float }

; Function Attrs: alwaysinline
define dso_local spir_func void @get2(%struct.A* sret %ret.ptr) local_unnamed_addr #0 {
  ; args are "ID" and "default value":
  %1 = tail call spir_func i32 @_Z20__spirv_SpecConstantii(i32 43, i32 0)
  %2 = tail call spir_func float @_Z20__spirv_SpecConstantif(i32 44, float 0.000000e+00)
  %ret = tail call spir_func %struct.A @_Z29__spirv_SpecConstantCompositeif(%1, %2)
  store %struct.A %ret, %struct.A* %ret.ptr
  ret void
}
```

the translator will generate `OpSpecConstant` SPIR-V instructions with proper
`SpecId` decorations:

```
              OpDecorate %i32 SpecId 42 ; ID
       %i32 = OpSpecConstant %int 0     ; Default value
              OpDecorate %i32 SpecId 43                            ; ID of the 1st member
              OpDecorate %float SpecId 44                          ; ID of the 2nd member
     %A.i32 = OpSpecConstant %int.type 0                           ; 1st member with default value
   %A.float = OpSpecConstant %float.type 0.0                       ; 2nd member with default value
    %struct = OpSpecConstantComposite %struct.type %A.i32 %A.float ; Composite doens't need IDs or default value
         %1 = OpTypeFunction %int

       %get = OpFunction %int None %1
         %2 = OpLabel
              OpReturnValue %i32
              OpFunctionEnd
         %1 = OpTypeFunction %struct.type

      %get2 = OpFunction %struct.type None %struct
         %2 = OpLabel
              OpReturnValue %struct
              OpFunctionEnd
```

#### DPC++ FE: Integration header generation

Information required for DPC++ RT in order to set value of a specialization
constant is provided by the compiler for each symbolic ID of specialization
constants. However, at DPC++ RT level the only information available is C++
typename, which was provided by user as a specialization constant name.

So, in order to connect those two together DPC++ FE emits additional section in
integration header, which allows to map C++ typename to symbolic ID of a
specialization constant:

```
// user code:
class MyIn32Constant;
cl::sycl::ONEAPI::experimental::spec_constant<int, MyInt32Const> i32(0);
// integration header:
template <> struct sycl::detail::SpecConstantInfo<::MyInt32Const> {
  static constexpr const char* getName() {
    return "_ZTS11MyInt32Const";
  }
};
```

NOTE: By using `__builtin_unique_stable_name` we could avoid modifying
integration header at all, but since the host part of the program can be
compiled with a third-party C++ 14-compatible compiler, which is unaware of the
clang-specific built-ins, it can result in build errors.

### DPC++ runtime

For each device binary compiler generates a map
\<Symbolic ID\> =\> \<list of spec constant descriptors\> ("ID map"). DPC++
runtime imports that map when loading device binaries.
It also maintains another map \<Spec const symbolic ID\> =\> \<its value\> 
("value map") per `sycl::program` object. The value map is updated upon
`program::set_spec_constant<IDType>(val)` calls from the app.

Before invoking JIT compilation of a program, the runtime "flushes"
specialization constants: it iterates through the value map and invokes

```
pi_result piextProgramSetSpecializationConstant(pi_program prog,
                                                pi_uint32 spec_id,
                                                size_t spec_size,
                                                const void *spec_value);
```

Plugin Interface function for descriptor of each property: `spec_id` and
`spec_size` are taken from the descriptor, `spec_value` is calculated based on
address of the specialization constant provided by user and `offset` field of
the descriptor.
