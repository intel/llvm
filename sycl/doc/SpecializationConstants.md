# Specialization constants

DPC++ implements this [proposal](https://github.com/codeplaysoftware/standards-proposals/blob/master/spec-constant/index.md)
with some restrictions. See this [document](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/SpecConstants/README.md) for more details.

#### Requirements:

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
class MyInt32Const;
...
  sycl::program p(q.get_context());
  sycl::ONEAPI::experimental::spec_constant<int32_t, MyInt32Const> i32 =
      p.set_spec_constant<MyInt32Const>(rt_val);
  p.build_with_kernel_type<MyKernel>();
  sycl::buffer<int, 1> buf(vec.data(), vec.size());

  q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.single_task<MyKernel>(
        p.get_kernel<MyKernel>(),
        [=]() {
          acc[0] = i32.get();
        });
  });
...
```

## Design

This section describes the basic design used to support spec constants of
primitive numeric types. POD types support is described further in the document.

#### Compiler

Key `spec_constant::get()` function implementation for the device code:

```
template <typename T, typename ID = T> class spec_constant {
...
public:
  T get() const { // explicit access.
#ifdef __SYCL_DEVICE_ONLY__
    const char *TName = __builtin_unique_stable_name(ID);
    return __sycl_getSpecConstantValue<T>(TName);
#else
    return Val;
#endif // __SYCL_DEVICE_ONLY__
  }
```

here `__builtin_unique_stable_name` is a compiler built-in used to translate
types to unique strings. `__sycl_getSpecConstantValue<T>` is an "intrinsic"
recognized by a special LLVM pass later.

Compilation and subsequent linkage of device code results in a number of
`__sycl_getSpecConstantValue` calls whose arguments are symbolic spec constant
IDs. Before generating a device binary, each linked device code LLVMIR
module undergoes processing by the sycl-post-link tool which can run LLVMIR
passes before passing the module onto the llvm-spirv translator.

There is a `SpecConstants` LLVMIR pass which
- assigns numeric IDs to the spec constants
- brings IR to the form expected by the llvm-spirv translator
- collects and provides \<Symbolic ID\> =\> \<numeric ID\> spec constant information
  to the sycl-post-link tool
Particularly, it translates intrinsic calls to the
`T __sycl_getSpecConstantValue*(const char *symbolic_id)` intrinsic into
calls to `T __spirv_SpecConstant(int ID, T default_val)` intrinsic known to
the llvm-spirv translator. Where `ID` is the numeric ID of the corresponding
spec constant, `default_val` is its default value which will be used if the
constant is not set at the runtime.

After this pass the sycl-post-link tool will output the
\<Symbolic ID\> =\> \<numeric ID\> spec constant mapping into a file for later
attaching this info to the device binary image via the offload wrapper tool as
a property set:

```
struct pi_device_binary_struct {
...
  // Array of preperty sets; e.g. specialization constants symbol-int ID map is
  // propagated to runtime with this mechanism.
  pi_device_binary_property_set PropertySetsBegin;
  pi_device_binary_property_set PropertySetsEnd;
};
```

SYCL runtime can then load and access info about particular spec constant using
its name as a key into the appropriate property set (named "SYCL/specialization
constants").

##### Ahead of time compilation

With AOT everything is simplified - the `SpecConstants` pass simply replaces
the `__sycl_getSpecConstantValue` calls with constants - default values of
the spec constant's type. No maps are generated, and SYCL program can't change
the value of a spec constant.

#### LLVM -> SPIR-V translation

Given the `__spirv_SpecConstant` intrinsic calls produced by the
`SpecConstants` pass:
```
; Function Attrs: alwaysinline
define dso_local spir_func i32 @get() local_unnamed_addr #0 {
  ; args are "ID" and "default value":
  %1 = tail call spir_func i32 @_Z20__spirv_SpecConstantii(i32 42, i32 0)
  ret i32 %1
}
```

the translator will generate `OpSpecConstant` SPIR-V instructions with proper
`SpecId` decorations:

```
              OpDecorate %i32 SpecId 42 ; ID
       %i32 = OpSpecConstant %int 0     ; Default value
         %1 = OpTypeFunction %int

       %get = OpFunction %int None %1
         %2 = OpLabel
              OpReturnValue %i32
              OpFunctionEnd
```

#### SYCL runtime

For each device binary compiler generates a map \<Symbolic ID\> =\> \<numeric ID\>
("ID map"). The SYCL runtime imports that map when loading device binaries.
It also maintains another map \<Spec const symbolic ID\> =\> \<its value\> 
("value map") per `sycl::program` object. The value map is updated upon
`program::set_spec_constant<IDType>(val)` calls from the app.

**_NOTE_**  `IDType` gets translated to the symbolic ID using the integration
header mechanism, similarly to kernel ID type. The reason why
`__builtin_unique_stable_name` is not used here is because this code is
compiled by the host compiler, which can be any C++ 14-compatible compiler
unaware of the clang-specific built-ins.

Before JIT-ing a program, the runtime "flushes" the spec constants: it iterates
through the value map and invokes the

```
pi_result piextProgramSetSpecializationConstant(pi_program prog,
                                                pi_uint32 spec_id,
                                                size_t spec_size,
                                                const void *spec_value);
```

Plugin Interface function for each entry, taking the `spec_id` from the ID map.

## "Plain Old Data" (POD) types support design

#### Source representation

Say, the POD type is

```
struct A {
  int x;
  float y;
};

struct POD {
  A a[2];
  cl::sycl::vec<int, 2> b;
};
```

and the user says

```
  POD gold{
    {
      { goldi, goldf },
      { goldi + 1, goldf + 1 },
    },
    { goldi, goldi }
  };

  cl::sycl::ONEAPI::experimental::spec_constant<POD, MyConst> sc =  program4.set_spec_constant<MyConst>(gold);
```

#### Compiler

##### The SpecConstants pass

The SpecConstants pass in the post-link will have the following IR as input
(`sret` conversion is omitted for clarity):

```
%struct.POD = type { [2 x %struct.A], <2 x i32> }
%struct.A = type { i32, float }

%spec_const = call %struct.POD __sycl_getCompositeSpecConstantValue<POD type mangling> ("MyConst_mangled")
```

`__sycl_getCompositeSpecConstantValue` is a new "intrinsic" (in addition to
`__sycl_getSpecConstantValue`) recognized by the `SpecConstants` pass, which
creates a value of a composite (of non-primitive type) specialization constant.
It does not need a default value, because its default value consists of default
values of its leaf specialization constants (see below).

`__sycl_getCompositeSpecConstantValue` will be replaced with a set of
`__spirv_SpecConstant` calls for each member of its return type plus one
`__spirv_SpecConstantComposite` to gather members back into a single composite.
If any composite member is another composite, then it will be also represented
by number of `__spirv_SpecConstant` plus one `__spirv_SpecConstantComposite`.

```
%gold_POD_A0_x = call i32 __spirv_SpecConstant(i32 10, i32 0)
%gold_POD_A0_y = call float __spirv_SpecConstant(i32 11, float 0)

%gold_POD_A0 = call %struct.A __spirv_SpecConstantComposite(i32 %gold_POD_A0_x, float %gold_POD_A0_y)

%gold_POD_A1_x = call i32 __spirv_SpecConstant(i32 12, i32 0)
%gold_POD_A1_y = call float __spirv_SpecConstant(i32 13, float 0)

%gold_POD_A1 = call %struct.A __spirv_SpecConstantComposite(i32 %gold_POD_A1_x, float %gold_POD_A1_y)

%gold_POD_A = call [2 x %struct.A] __spirv_SpecConstantComposite(%struct.A %gold_POD_A0, %struct.A %gold_POD_A1)

%gold_POD_b0 = call i32 __spirv_SpecConstant(i32 14, i32 0)
%gold_POD_b1 = call i32 __spirv_SpecConstant(i32 15, i32 0)
%gold_POD_b = call <2 x i32> __spirv_SpecConstant(i32 %gold_POD_b0, i32 %gold_POD_b1)

%gold = call %struct.POD __spirv_SpecConstantComposite([2 x %struct.A] %gold_POD_A, <2 x i32> %gold_POD_b)

```

Spec ID for the composite spec constant is not needed, as runtime will never use
it - it will use IDs of the leaves instead, which are being assigned by the
`SpecConstants` pass during replacement of SYCL intrinsics with SPIR-V
intrinsics.
Besides, the SPIR-V specification does not allow `SpecID` decoration for
composite spec constants, because its defined by its members instead.

`__spirv_SpecConstantComposite` is a new SPIR-V intrinsic, which represents
composite specialization constant. Its arguments are LLVM IR values
corresponding to elements of the composite constant.

##### LLVM -> SPIR-V translation

Given the `__spirv_SpecConstantComposite` intrinsic calls produced by the
`SpecConstants` pass:
```

%struct.A = type { i32, float }

; Function Attrs: alwaysinline
define dso_local spir_func void @get(%struct.A* sret %ret.ptr) local_unnamed_addr #0 {
  ; args are "ID" and "default value":
  %1 = tail call spir_func i32 @_Z20__spirv_SpecConstantii(i32 42, i32 0)
  %2 = tail call spir_func float @_Z20__spirv_SpecConstantif(i32 43, float 0.000000e+00)
  %ret = tail call spir_func %struct.A @_Z29__spirv_SpecConstantCompositeif(%1, %2)
  store %struct.A %ret, %struct.A* %ret.ptr
  ret void
}
```

the translator will generate `OpSpecConstant` and `OpSpecConstantComposite`
SPIR-V instructions with proper `SpecId` decorations:

```
              OpDecorate %i32 SpecId 42                        ; ID of the 1st member
              OpDecorate %float SpecId 43                      ; ID of the 2nd member
       %i32 = OpSpecConstant %int.type 0                       ; 1st member with default value
     %float = OpSpecConstant %float.type 0.0                   ; 2nd member with default value
    %struct = OpSpecConstantComposite %struct.type %i32 %float ; Composite doens't need IDs or default value
         %1 = OpTypeFunction %struct.type

       %get = OpFunction %struct.type None %1
         %2 = OpLabel
              OpReturnValue %struct
              OpFunctionEnd
```

##### The post-link tool changes

For composite specialization constants the post link tool will additionally
generate linearized list of \<leaf spec ID,offset,size\> tuples (descriptors),
where each tuple describes a leaf field, and store it together with the
existing meta-information associated with the specialization constants and
passed to the runtime. Also, for a composite specialization constant there is
no ID map entry within the meta information, and the composite constant is
referenced by its symbolic ID. For example:

```
MyConst_mangled [10,0,4],[11,4,4],[12,8,4],[13,12,4],[14,16,4]
```

This tuple is needed, because at SYCL runtime level, composite constants are set
by user as a byte array and we have to break it down to the leaf members of the
composite and set a value for each leaf as for a separate scalar specialization
constant. Each tuple contains the following data:
- ID of composite constant leaf, i.e. ID of a scalar specialization constant
- Offset from the beginning of composite, which points to the location of a
  scalar value within the composite, i.e. the position where scalar
  specialization constant resides within the byte array supplied by the user
- Size of the scalar specialization constant

#### SYCL runtime

First, when the runtime loads a binary it gets access to specialization
constant information. So the mapping from a composite spec constant name to
its constituents (descriptors of leaf fields) generated by the post-link tool
will be available.

Now, when the program invokes `program4.set_spec_constant<MyConst>(gold)`,
 SYCL runtime converts the call arguments (template and actual) to the following
 pair of datums:
 - the constant name - "MyConst_mangled"
 - the byte array representing the value of the constant (`gold` value)
 
Then the runtime fetches the sequence of leaf field descriptors (primitive
constituents) of the composite constant and iterates through each pair invoking
`piextProgramSetSpecializationConstant` for each. The ID of the constant is
taken from the sequence, the value - from the byte array obtained for the
`gold`.
