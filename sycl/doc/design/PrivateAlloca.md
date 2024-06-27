# `sycl_ext_oneapi_private_alloca`

SYCL does not allow dynamically sized private memory allocations. The
[`sycl_ext_oneapi_private_alloca`](../extensions/experimental/sycl_ext_oneapi_private_alloca.asciidoc)
extension is a way to lift this restriction by allowing memory allocations whose
length is given by a SYCL specialization constant.

## Use-case example

```c++
#include <sycl/sycl.hpp>

constexpr sycl::specialization_id<std::size_t> size(1);

namespace syclex = sycl::ext::oneapi::experimental;

class Kernel;

SYCL_EXTERNAL void impl(const float *in, float *out,
                        sycl::span<float> ptr);

void run(sycl::queue q, const float *in, float *out, size_t n) {
  q.submit([&](sycl::handler &h) {
    h.set_specialization_constant<size>(n);
    h.parallel_for<Kernel>(n, [=](sycl::id<1> i, sycl::kernel_handler kh) {
      sycl::span<float> tmp{
          syclex::private_alloca<float, size, sycl::access::decorated::no>(kh).get_raw(),
          kh.get_specialization_constant<size>()};
      impl(in, out, tmp);
    });
  });
}
```

In this use-case, every work-item allocates a private memory region capable of
hosting `kh.get_specialization_constant<size>()` elements of type `float`. This
might be used for performance improvements in some algorithms, e.g., needing
more than one iteration on a sub-sequence of the input. However, as the length
of this sub-sequence is an input argument, this can only be achieved via
`sycl_ext_oneapi_private_alloca`. Combining
`sycl::handler::set_specialization_constant<size>` and calling `private_alloca`,
we can get a dynamically sized memory allocation. Note `size` is guaranteed to
be constant during kernel execution.

## Design

`sycl_ext_oneapi_private_alloca` is currently only supported on targets with
native spec constants support when AOT compilation is not used.

For non-SPIR-V targets, aspect checking is used to check
`sycl_ext_oneapi_private_alloca` support. In case of AOT compilation, a
compile-time error is produced as this kind of checks cannot be performed at
runtime via aspects.

### Usage in SYCL host code

Calling either of the functions defined in this extension in host code results
in a synchronous exception with the `errc::feature_not_supported` error code.

```c++
#ifdef __SYCL_DEVICE_ONLY__
...
#else
template <typename ElementType, auto &SizeSpecName,
          access::decorated DecorateAddress>
private_ptr<ElementType, DecorateAddress> private_alloca(kernel_handler &kh) {
  throw sycl::exception(sycl::errc::feature_not_supported,
                        "Images are not supported by this device.");
}
#endif
```

### New `__builtin_intel_sycl_alloca` and `__builtin_intel_sycl_alloca_with_align` builtins

`private_alloca` and `aligned_private_alloca` APIs are defined as builtin
aliases of `__builtin_intel_sycl_alloca` and
`__builtin_intel_sycl_alloca_with_align` respectively using the
`clang::builtin_alias` attribute. This way, calls to these functions are handled
as calls to the builtins in the frontend. These builtins cannot be called
directly, only via their aliases defined in the SYCL headers.

```c++
template <typename ElementType, auto &SizeSpecName,
          access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca)
[[__sycl_detail__::__uses_aspects__(aspect::ext_oneapi_private_alloca)]] private_ptr<
    ElementType, DecorateAddress> private_alloca(kernel_handler &kh);

template <typename ElementType, std::size_t Alignment,
          auto &SizeSpecName, access::decorated DecorateAddress>
__SYCL_BUILTIN_ALIAS(__builtin_intel_sycl_alloca_with_align)
[[__sycl_detail__::__uses_aspects__(aspect::ext_oneapi_private_alloca)]] private_ptr<
    ElementType, DecorateAddress> aligned_private_alloca(kernel_handler &kh);
```

As builtins cannot be passed template arguments,
`Sema::CheckIntelSYCLAllocaBuiltinFunctionCall` and
`CodeGenFunction::EmitIntelSYCLAllocaBuiltin` do not use the builtin for
checking or code generation. Instead, the original function declaration is
queried and used.

Following `__builtin_alloca_with_align`, `Alignment` must be lower than
`std::numeric_limits<int32_t>::max() / 8`.

Note using `clang::builtin_alias` required modification to code checking this
attribute to add SYCL to the list of contexts in which you can use
it. Implementation of this extension upstream would need to port these changes
upstream or modify code generation and sema of `private_alloca` and
`aligned_private_alloca` APIs.

### `llvm.sycl.alloca.*` intrinsic

The builtins mentioned in the previous section are represented via the new
`llvm.sycl.alloca` intrinsic in code generation. This intrinsic receives three
arguments encoding the specialization constant used as array length,
corresponding to the arguments received by builtins implementing SYCL 2020
specialization constants; a type hint argument encoding the allocation element
type, and the required alignment, which must be an immediate argument. Note
`sycl-post-link` usage of `__spirv_SpecConstant` to represent specialization
constant queries is preserved, as we can reuse most of the handling of
specialization constant builtins defined in [the corresponding design
document](./SYCL2020-SpecializationConstants.md).

```llvm
declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.f32(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), float, i64)
declare ptr @llvm.sycl.alloca.p0.p4.p4.p4.f64(ptr addrspace(4), ptr addrspace(4), ptr addrspace(4), double, i64)
```

The alignment argument is set to `alignof(ElementType)` for `private_alloca`;
`Alignment` is used for `aligned_private_alloca`.

The `private_alloca` call in the use-case above is compiled to the following
LLVM IR:

```llvm
@_ZL4size = internal addrspace(1) constant %"class.sycl::_V1::specialization_id" { i64 1 }, align 8
@__usid_str.1 = private unnamed_addr constant [31 x i8] c"uid2c9b8e1a387f5dba____ZL4size\00", align 1
...
%alloca.i = tail call ptr @llvm.sycl.alloca.p0.p4.p4.p4.f32(ptr addrspace(4) addrspacecast (ptr @__usid_str.1 to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @_ZL4size to ptr addrspace(4)), ptr addrspace(4) null, float 0.000000e+00, i64 4)
```

**Note:** the third argument is set to `null` for now. That argument will be set
to the `RTBuffer` for the specialization constants, i.e., a pointer to the
corresponding member of the input `sycl::kernel_handler`. This change will be
needed to support this extension in non-SPIR-V targets, i.e., in targets with
emulated specialization constants.

`llvm.sycl.alloca` is handled in `sycl-post-link`.

### `sycl-post-link`

`sycl-post-link` was modified to handle the intrinsic above in addition to SYCL
2020 specialization constants builtins. These are replaced by an `alloca`
instruction of the element type given by the intrinsic type hint and the size
given by the input specialization constant. Code used to obtain a specialization
constant value is reused, as the intrinsic receives the same three arguments as
the regular builtins.

This way, the `-spec-const` pass (using the `native` option) transforms the LLVM
IR code above into:

```llvm
%size = call i64 @_Z20__spirv_SpecConstantix(i32 0, i64 1)
%alloca.i = alloca float, i64 %size, align 4
```

The builtin in conjunction with the `alloca` instruction are handled by the
LLVM-SPIR-V translator to generate valid SPIR-V code.

### LLVM-SPIR-V translator

The LLVM-SPIR-V  translator can  handle `alloca` instructions  as the  one above
generating  only standard  SPIR-V  operations.  SPIR-V array  types  can have  a
specialization constant length,  so the running example is  translated to SPIR-V
as follows:

```
                 Decorate %size SpecId 0
       %sizety = OpTypeInt 64 0
      %floatty = OpTypeFloat 32
         %size = SpecConstant %sizety 1
        %arrty = OpTypeArray %floatty %size
   %floatptrty = OpTypePointer Function %floatty
     %arrptrty = OpTypePointer Function %arrty
%genfloatptrty = OpTypePointer Generic %floatty
...
       %alloca = OpVariable %arrptrty Function
      %bitcast = OpBitcast %floatptrty %alloca
```

When passed a specialization constant as size, a single `alloca` instruction is
represented as an `OpVariable` operation of an array type of specialization
constant size and an `OpBitcast` operation to cast the variable to the required
pointer type, i.e., to a pointer type of the array element type.
