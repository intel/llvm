# Specialization constants

DPC++ implements the [proposal](https://github.com/codeplaysoftware/standards-proposals/blob/master/spec-constant/index.md)
by Codeplay with some restrictions. See this [document](https://github.com/intel/llvm/pull/2503) for more details.

#### Requirements:
- must work with separate compilation and linking
- must support AOT compilation

Implementaion is based on SPIRV specialization constants. But there is one
important difference between SYCL and SPIRV: in SYCL speciazation constants are
identified by a type ID which is mapped to a symbolic name, in SPIRV - by an
ordinal number. This complicates the design, as the compiler
1) needs to propagate symbolic =\> numeric ID correspondence to the runtime
2) can assign numeric IDs only when linking due to the separate compilation

Simple source code example:

```cpp
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
```cpp
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
IDs. Before generating the a device binary, each linked device code LLVMIR
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

```cpp
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

#### LLVMIR-SPIRV translator

Given the `__spirv_SpecConstant` intrinsic calls produced by the
`SpecConstants` pass:
```cpp
; Function Attrs: alwaysinline
define dso_local spir_func i32 @get() local_unnamed_addr #0 {
  ; args are "ID" and "default value":
  %1 = tail call spir_func i32 @_Z20__spirv_SpecConstantii(i32 42, i32 0)
  ret i32 %1
}
```
the translator will generate `OpSpecConstant` SPIRV instructions with proper
`SpecId` decorations:
```cpp
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

***NOTE**: `IDType` gets translated to the symbolic ID using the integration
header mechanism, similarly to kernel ID type. The reason why
`__builtin_unique_stable_name` is not used here is because this code is
compiled by the host compiler, which can be any C++ 14-compatible compiler
unaware of the clang-specific built-ins.*

Before JIT-ing a program, the runtime "flushes" the spec constants: it iterates
through the value map and invokes the
```cpp
pi_result piextProgramSetSpecializationConstant(pi_program prog,
                                                pi_uint32 spec_id,
                                                size_t spec_size,
                                                const void *spec_value);
```

Plugin Interface function for each entry, taking the `spec_id` from the ID map.

## "Plain Old Data" (POD) types support design

#### Source representation

Say, the POD type is
```cpp
struct A {
  int x;
  float y;
};

struct POD {
  A a[2];
  int b;
};
```
and the user says
```cpp
  POD gold{
    {
      { goldi, goldf },
      { goldi + 1, goldf + 1 },
    },
    goldi
  };

  cl::sycl::ONEAPI::experimental::spec_constant<POD, MyConst> sc =  program4.set_spec_constant<MyConst>(gold);
```
#### Compiler
##### The SpecConstant pass changes
 - The SpecConstants pass in the post-link will have the following IR as input (`sret` conversion is omitted for clarity):
```cpp
  %spec_const = call %struct.POD __sycl_getCompositeSpecConstantValue<mangling for POD type template specialization ("MyConst_mangled")
```
where `__sycl_getCompositeSpecConstantValue` is a new "intrinsic"
 (in addition to `__sycl_getSpecConstantValue`) recognized by SpecConstants pass,
 which creates a value of a composite (of non-primitive type) specialization constant.
 It does not need a default value, because its default value consists of default
 valued of its leaf specialization constants (see below).

 - after spec constant enumeration (symbolic -\> int ID translation), the SC pass
 will handle the `__sycl_getCompositeSpecConstantValue`. Knowning the composite
 specialization constant's type (`%struct.POD`), the pass will traverse its leaf
 fields and generate 5 "primitive" spec constants using already existing SPIRV intrinsic:
```cpp
%gold_POD_a0x = call i32 __spirv_SpecConstant(i32 10, i32 0)
%gold_POD_a0y = call float __spirv_SpecConstant(i32 11, float 0)
%gold_POD_a1x = call i32 __spirv_SpecConstant(i32 12, i32 0)
%gold_POD_a1y = call float __spirv_SpecConstant(i32 13, float 0)
%gold_POD_b = call i32 __spirv_SpecConstant(i32 14, i32 0)
```
And 1 "composite"
```cpp
  %gold_POD = call %struct.POD __spirvCompositeSpecConstant<POD mangling>(10, 11, 12, 13, 14, 15)
```
where `__spirvCompositeSpecConstant<type mangling>` is a new SPIRV intrinsic which
 represents creation of a composite specialization constant. Its arguments are spec
 constant IDs corresponding to the leaf fields of the POD type of the constant.
 ID is not needed, as runtime will never use it - it will use IDs of the leaves instead.
 Yet SPIRV does not allow IDs for composite spec constants.

##### The post-link tool changes

For composite specialization constants the post link tool will additionally
generate linearized list of \<leaf spec ID,type,offset,size\> tuples (descriptors),
where each tuple describes a leaf field, and store it together with the
existing meta-information associated with the specialization constants and
passed to the runtime. Also, for a composite specialization constant the will be
no ID map entry within the meta information, and the composite constant will
referenced by its symbolic ID. For example:
```cpp
MyConst_mangled [10,int,0,4],[10,float,4,4],[10,int,8,4],[10,float,12,4],[10,int,16,4]
```

#### LLVMIR-\>SPIRV translator

The translator aims to create the following code (pseudo-code)
```cpp
%gold_POD_a0x = OpSpecConstant(0)    [SpecId = 10]
%gold_POD_a0y = OpSpecConstant(0.0f) [SpecId = 11]
%gold_POD_a1x = OpSpecConstant(0)    [SpecId = 12]
%gold_POD_a1y = OpSpecConstant(0.0f) [SpecId = 13]
%gold_POD_b   = OpSpecConstant(0)    [SpecId = 14]

%gold_POD_a0 = OpSpecConstantComposite(
  %gold_POD_a0x // gold.a[0].x
  %gold_POD_a0y // gold.a[0].y
)

%gold_POD_a1 = OpSpecConstantComposite(
  %gold_POD_a1x // gold.a[1].x
  %gold_POD_a1y // gold.a[1].y
)

%gold_POD = OpSpecConstantComposite(
  %gold_POD_a0,
  %gold_POD_a1,
  %gold_POD_b   // gold.b
}
```
 - `OpSpecConstant` operations will be created using already existing mechanism for
 the primitive spec constants.
 - Then the translator will handle `__spirvCompositeSpecConstant*` intrinsic. It will
 recursively traverse the spec constant type structure in parallel with the argument
 list - which is a list of primitive spec constant operation IDs (not their SpecIds!).
 When traversing, it will create all the intermediate OpSpecConstantComposite
 operations as well as the root one (`%gold_POD`) using simple depth-first tree
 traversal with stack. This requires mapping from SpecId decoration number to \<id\> of the corresponding OpSpecConstant operation, but this should be pretty straightforward.

#### SYCL runtime

First, when the runtime loads a binary it gets access to specialization
constant information. So this mapping from a composite spec constant name to
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
