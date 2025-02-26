# Specialization constants

Specialization constants are implemented in accordance with how they are defined
by SYCL 2020 specification: [SYCL registry][sycl-registry],
[direct link to the specification][sycl-2020-spec].

[sycl-registry]: https://www.khronos.org/registry/SYCL/
[sycl-2020-spec]: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html

> Specialization constants represent constants whose values can be set
> dynamically during execution of the SYCL application. The values of these
> constants are fixed when a SYCL kernel function is invoked, and they do not
> change during the execution of the kernel. However, the application is able to
> set a new value for a specialization constants each time a kernel is invoked,
> so the values can be tuned differently for each invocation.
>
> [Section 4.9.5 Specialization constants][sycl-2020-4-9-5]

[sycl-2020-4-9-5]: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_specialization_constants

Example usage:

```
#include <sycl/sycl.hpp>
using namespace sycl;

using coeff_t = std::array<std::array<float, 3>, 3>;

// Read coefficients from somewhere.
coeff_t get_coefficients();

// Identify the specialization constant.
constexpr specialization_id<coeff_t> coeff_id;

void do_conv(buffer<float, 2> in, buffer<float, 2> out) {
  queue myQueue;

  myQueue.submit([&](handler &cgh) {
    accessor in_acc { in, cgh, read_only };
    accessor out_acc { out, cgh, write_only };

    // Set the coefficient of the convolution as constant.
    // This will build a specific kernel the coefficient available as literals.
    cgh.set_specialization_constant<coeff_id>(get_coefficients());

    cgh.parallel_for<class Convolution>(
        in.get_range(), [=](item<2> item_id, kernel_handler h) {
          float acc = 0;
          coeff_t coeff = h.get_specialization_constant<coeff_id>();
          for (int i = -1; i <= 1; i++) {
            if (item_id[0] + i < 0 || item_id[0] + i >= in_acc.get_range()[0])
              continue;
            for (int j = -1; j <= 1; j++) {
              if (item_id[1] + j < 0 || item_id[1] + j >= in_acc.get_range()[1])
                continue;
              // The underlying JIT can see all the values of the array returned
              // by coeff.get().
              acc += coeff[i + 1][j + 1] *
                     in_acc[item_id[0] + i][item_id[1] + j];
            }
          }
          out_acc[item_id] = acc;
        });
  });

  myQueue.wait();
}
```

## Design objectives

SYCL 2020 [defines specialization constant][sycl-2020-spec-constant-glossary]
as:

> A constant variable where the value is not known until compilation of the
> SYCL kernel function.
>
> [Glossary][sycl-2020-glossary]

[sycl-2020-spec-constant-glossary]: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#specialization-constant
[sycl-2020-glossary]: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#glossary

And implementation is based on [SPIR-V speficiation][spirv-spec] support
for [Specialization][spirv-specialization]. However, the specification also
states the following:

> It is expected that many implementations will use an intermediate language
> representation ... such as SPIR-V, and the intermediate language will have
> native support for specialization constants. However, implementations that do
> not have such native support must still support specialization constants in
> some other way.
>
> [Section 4.11.12.2. Specialization constant support][sycl-2020-4-11-12-2]

[spirv-spec]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html
[spirv-specialization]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#SpecializationSection
[sycl-2020-4-11-12-2]: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#_specialization_constant_support

Having that said, the following should be implemented:

1. We need to ensure that in generated SPIR-V, calls to
`get_specialization_constant` are replaced with corresponding instructions for
referencing SPIR-V specialization constants.

2. SYCL provides a mechanism to specify default values of specialization
constants, which should be reflected in the generated SPIR-V. This part is
especially tricky, because this happens in host part of the SYCL program, which
means that without special handling it won't even be visible to the device
compiler.

3. We need to ensure that DPC++ RT properly sets specialization constants used
in the program: SYCL spec uses non-type template parameters to identify
specialization constants in the program, while at SPIR-V and OpenCL levels, each
specialization constant is defined by its numerical ID, which means that we
need to maintain some mapping from SYCL identifiers to a numeric identifiers in
order to be able to set specialization constants. Moreover, at SPIR-V level
composite specialization constants do not have separate ID and can only be set
by setting value to each member of a composite, which means that we have `1:n`
mapping between SYCL identifiers and numeric IDs of specialization constants.

4. When AOT compilation is used or the target device does not use SPIR-V as the
device code format (for example, CUDA device, where NVPTX intermediate
representation is used), we need to somehow emulate support for specialization
constants.

## Design

As stated above, native specialization constants support is based on
corresponding SPIR-V functionality, while emulation is implemented through
transforming specialization constants into kernel arguments.

In DPC++ Headers/DPC++ RT we don't know a lot of necessary information about
specialization constants, like: which numeric ID is used for particular
specialization constant (since we support `SYCL_EXTERNAL`, those IDs can only
be allocated by the compiler during link stage) or which kernel argument is used
to pass particular specialization constant (because they are not explicitly
captured by SYCL kernel functions and regular mechanism for kernel arguments
handling can't be used here).

Therefore, we can't have headers-only implementation and the crucial part of
design is how to organize mapping mechanism between SYCL identifiers for
specialization constants (`specialization_id`s) and low-level identifiers
(numeric IDs in SPIR-V or information about corresponding kernel arguments).

That mapping mechanism is particularly tricky, because of some additional
complexity coming from SYCL 2020 specification:
- `specialization_id` variables, which are used as specialization constant
  identifiers (being non-type template parameters of some methods) can't be
  forward-declared in general case (for example, if defined as `static`), which
  means that we can't use integration header to attach some information to them
  through some C++ templates tricks (like it is done for regular kernel
  arguments or kernel names, for example).
- they also can be declared as `static` or just non-`inline` `constexpr`, which
  means that they have internal linkage and can't be referenced from other
  translation units, which means that we can't for example create a new
  translation unit which contains some mapping from `specialization_id` address
  to some desired info.

Based on those limitations, the following mapping design is proposed:
- DPC++ RT uses special function:
  ```
  namespace detail {
    template<auto &SpecName>
    inline const char *get_spec_constant_symbolic_ID();
  }
  ```
  Which is only declared, but not defined in there and used to retrieve required
  information like numeric ID of a specialization constant.
- Definition of that function template are provided by DPC++ FE in form of
  _integration footer_: the compiler generates a piece of C++ code which is
  injected at the end of a translation unit:
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

  Those symbolic IDs are used to identify device image properties corresponding
  to those specialization constants, which store additional information (like
  numeric SPIR-V ID of a constant) needed for DPC++ RT.
- That integration footer is automatically appended by the compiler at the end
  of user-provided translation unit by driver.

Another significant part of the design is how specialization constants support
is emulated: as briefly mentioned before, the general approach is to transform
specialization constants into kernel arguments. In fact all specialization
constants used within a program are bundled together and stored into a single
buffer, which is passed as implicit kernel argument. The layout of that buffer
is well-defined and known to both the compiler and the runtime, so when user
sets the value of a specialization constant, that value is being copied into
particular place within that buffer and once the constant is requested in
device code, the compiler generates a load from the same place of the buffer.

Summarizing, overall design looks like:

DPC++ Headers provide special markup, which used by the compiler to detect
presence of specialization constants and properly handle them.

DPC++ FE handles `kernel_handler` SYCL kernel function argument, creates
additional kernel argument to pass specialization constants through buffer if
necessary (i.e. if native support is not available) and generates integration
footer.

`sycl-post-link` transforms device code to either generate proper SPIR-V
Friendly IR with specialization constants (when native support is available) or
to generate correct access to corresponding kernel argument (which are used
when native support is not available); also the tool generates some device image
properties with all information needed for DPC++ RT (like which numeric SPIR-V
ID was assigned to which symbolic ID).

With help of `clang-offload-wrapper` tool, those device image properties are
embedded into the application together with device code and used by DPC++ RT
while handling specialization constants during application execution: it either
calls corresponding UR API to set a value of a specialization constant or it
fills a special buffer with values of specialization constants and passes it as
kernel argument to emulate support of specialization constants.

Sections below describe each component in more details.

### DPC++ Headers

DPC++ Headers provide required definitions of `specialization_id` and
`kernel_handler` classes as well as of many other classes and methods.

`kernel_handler::get_specialization_constant` method, which provides an access
to specialization constants within device code implements an interface between
DPC++ Headers and the compiler (`sycl-post-link` tool): it contains a special
markup, which allows the compiler to detect specialization constants in the
device code and properly handle them.

```
namespace sycl {
template<typename T>
T __sycl_getScalar2020SpecConstantValue<T>(const char *SymbolicID, const void *DefaultValue, const void *RTBuffer);
template<typename T>
T __sycl_getComposite2020SpecConstantValue<T>(const char *SymbolicID, const void *DefaultValue, const void *RTBuffer);

class kernel_handler {
public:
  template<auto& SpecName>
  typename std::remove_reference_t<decltype(SpecName)>::type get_specialization_constant() {
#ifdef __SYCL_DEVICE_ONLY__
    return get_on_device<SpecName>();
#else
    // some fallback implementation in case this code is launched on host
#endif __SYCL_DEVICE_ONLY__
  }

private:
#ifdef __SYCL_DEVICE_ONLY__
  template<auto &SpecName, typename T = std::remove_reference_t<decltype(SpecName)>::type>
  // enable_if T is a scalar type
  T get_on_device() {
    auto ID = __builtin_sycl_unique_id(SpecName);
    return __sycl_getScalar2020SpecConstantValue<T>(ID, &SpecName, Ptr);
  }

  template<auto &SpecName, typename T = std::remove_reference_t<decltype(SpecName)>::type>
  // enable_if T is a composite type
  T get_on_device() {
    auto ID = __builtin_sycl_unique_id(SpecName);
    return __sycl_getComposite2020SpecConstantValue<T>(ID, &SpecName, Ptr);
  }
#endif // __SYCL_DEVICE_ONLY__

  byte *Ptr = nullptr;
};

} // namespace sycl
```

Here `__builtin_sycl_unique_id` is a new compiler built-in which is supposed to
generate unique symbolic IDs for specialization constants.

`__sycl_getScalar2020SpecConstantValue<T>` and
`__sycl_getComposite2020SpecConstant<T>` are functions with special names - they
are declared in the headers but never defined. Calls to them are recognized by
a special LLVM pass later and this is aforementioned special markup required for
the compiler.

Those functions accept three parameters:
1. Symbolic ID of specialization constant. This must be a constant string, which
   will be used by the compiler to uniquely identify the specialization
   constant. Device image properties generated by the compiler will use that
   string as well to attach additional data to the constant.

2. Default value of the specialization constant.
   It is expected that at LLVM IR level the argument will contain a pointer to
   a global variable with the initializer, which should be used as the default
   value of the specialization constants.

3. Pointer to a buffer, which will be used if native specialization constants
   are not available. This pointer is described later in the section
   corresponding to emulation of specialization constants.

Compilation and subsequent linkage of the device code results in a number of
`__sycl_getScalar2020SpecConstantValue` and
`__sycl_getComposite2020SpecConstantValue` calls. Before generating a device
binary, each linked device code LLVM IR module undergoes processing by
`sycl-post-link` tool which can run LLVM IR passes before passing the module
onto the SPIR-V translator.

### DPC++ Compiler: sycl-post-link tool

As it is stated above, the only place where we can properly handle
specialization constants is somewhere during or after linking device code from
different translation units, so it happens in `sycl-post-link` tool.

There is a `SpecConstantsPass` LLVM IR pass which:
1. Assigns numeric IDs to specialization constants found in the linked module.
2. Transforms IR to either:
   1. The form expected by the SPIR-V translator (format of the
     expected IR is covered in "Transformation of LLVM IR to SPIR-V friendly IR
     form" section).
   2. The form which is used for emulating specialization constants.
3. Collects and provides \<Symbolic ID\> =\> \<numeric IDs + additional info\>
   mapping, which is later being used by DPC++ RT to set specialization constant
   values provided by user (section "Collecting spec constants info and
   communicating it to DPC++ RT" provides more info on that)

#### 1. Assignment of numeric IDs to specialization constants

This task is achieved by maintaining a map, which holds a list of numeric IDs
for each encountered symbolic ID of a specialization constant. Those IDs are
used to identify the specialization constants at SPIR-V level.

As noted above one symbolic ID can have several numeric IDs assigned to it -
such 1:N mapping comes from the fact that at SPIR-V level, composite
specialization constants don't have dedicated IDs and they are being identified
and specialized through their scalar leafs and corresponding numeric IDs.

For example, the following code:
```
struct Nested {
  float a, b;
};
struct A {
  int x;
  Nested n;
};

constexpr specialization_id<int> id_int;
constexpr specialization_id<A> id_A;
// ...
  [=](kernel_handler h) {
    h.get_specialization_constant<id_int>();
    h.get_specialization_constant<id_A>();
  }
```

Will result in the following numeric IDs assignment:
```
// since `id_int` is a simple arithmetic specialization constant, we only
// have a single numeric ID associated with its symbolic ID
unique_symbolic_id_for_id_int -> { 0 }
// `id_A` is a composite with three leafs (scalar members, including ones
// located in nested composite types), which results in three numeric IDs
// associated with the same symbolic ID
unique_symbolic_id_for_id_A -> { 1, 2, 3 }
```

As it is shown in the example above, if a composite specialization constant
contains another composite within it, that nested composite is also being
"flattened" and its leafs are considered to be leafs of the parent
specialization constants. This done by depth-first search through the composite
elements.

#### 2.1 Transformation of LLVM IR to SPIR-V friendly IR form

SPIR-V friendly IR form is a special representation of LLVM IR, where some
function are named in particular way in order to be recognizable by the SPIR-V
translator to convert them into corresponding SPIR-V instructions later.
The format is documented [here][spirv-friendly-ir].

[spirv-friendly-ir]: https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/master/docs/SPIRVRepresentationInLLVM.rst

For specialization constant, we need to generate the following constructs:
```
template<typename T> // T is arithmetic type
T __spirv_SpecConstant(int numericID, T default_value);

template<typename T, typename... Elements> // T is composite type,
// Elements are arithmetic or composite types
T __spirv_SpecConstantComposite(Elements... elements);
```

Particularly, `SpecConstantsPass` translates calls to the
`T __sycl_getScalar2020SpecConstantValue(const char *SymbolicID, const void *DefaultValue, const char *RTBuffer)`
intrinsic into calls to `T __spirv_SpecConstant(int ID, T default_val)`.
And for `T __sycl_getComposite2020SpecConstantValue(const char *SybmolicID, const void *DefaultValue, const char *RTBuffer)`
it generates number of `T __spirv_SpecConstant(int ID, T default_val)` calls for
each leaf of the composite type, plus number of
`T __spirv_SpecConstantComposite(Elements... elements)` for each composite type
(including the outermost one).

Example of LLVM IR transformation can be found below, input LLVM IR:
```
%struct.POD = type { [2 x %struct.A], <2 x i32> }
%struct.A = type { i32, float }

@gold_scalar_default = global %class.specialization_id { i32 42 }
@gold_default = global %class.specialization_id { %struct.POD { [2 x %struct.A] [%struct.A { i32 1, float 2.000000e+00 }, %struct.A { i32 2, float 3.000000e+00 }], <2 x i32> <i32 44, i32 44> } }


; the second argument of intrinsics below are simplified a bit
; in real-life LLVM IR it looks like:
;   i8* bitcast (%class.specialization_id* @gold_scalar_default to i8*
%gold_scalar = call i32 __sycl_getScalar2020SpecConstantValue<int type mangling> ("gold_scalar_identifier", @gold_scalar_default, i8* %buffer)
%gold = call %struct.POD __sycl_getComposite2020SpecConstantValue<POD type mangling> ("gold_identifier", @gold_default, i8* %default)
```

LLVM IR generated by `SpecConstantsPass`:
```
%gold_scalar = call i32 __spirv_SpecConstant(i32 0, i32 42)

%gold_POD_A0_x = call i32 __spirv_SpecConstant(i32 1, i32 1)
%gold_POD_A0_y = call float __spirv_SpecConstant(i32 2, float 2.000000e+00)

%gold_POD_A0 = call %struct.A __spirv_SpecConstantComposite(i32 %gold_POD_A0_x, float %gold_POD_A0_y)

%gold_POD_A1_x = call i32 __spirv_SpecConstant(i32 3, i32 2)
%gold_POD_A1_y = call float __spirv_SpecConstant(i32 4, float 3.000000e+00)

%gold_POD_A1 = call %struct.A __spirv_SpecConstantComposite(i32 %gold_POD_A1_x, float %gold_POD_A1_y)

%gold_POD_A = call [2 x %struct.A] __spirv_SpecConstantComposite(%struct.A %gold_POD_A0, %struct.A %gold_POD_A1)

%gold_POD_b0 = call i32 __spirv_SpecConstant(i32 4, i32 44)
%gold_POD_b1 = call i32 __spirv_SpecConstant(i32 6, i32 44)
%gold_POD_b = call <2 x i32> __spirv_SpecConstant(i32 %gold_POD_b0, i32 %gold_POD_b1)

%gold = call %struct.POD __spirv_SpecConstantComposite([2 x %struct.A] %gold_POD_A, <2 x i32> %gold_POD_b)
```

#### 2.2 Transformation of LLVM IR for emulating specialization constants

In case we are not targeting SPIR-V, we don't have a native support for
specialization constants and have to emulate them somehow. As stated above, it
is done by converting specialization constants into kernel arguments: they all
bundled together and put into a single buffer.

`SpecConstantsPass` should generate proper accesses to that buffer when
specialization constants are used: this is done by replacing special
`__sycl_getScalar2020SpecConstantValue` and
`__sycl_getComposite2020SpecConstantValue` functions with accesses to their
third argument, which contains a pointer to the buffer with values of all
specialization constants. That access looks like a sequence of the following
LLVM IR instruction `getelementptr` from the buffer pointer by calculated,
offset, then `bitcast` to pointer to proper return type (because the buffer
pointer is just an "untyped" `i8 *`) and `load`. An example of that LLVM IR:
```
; an example for:
; constexpr specialization_id<double> id_double;
;   [=](kernel_handler h) {
;     h.get_specialization_constant<id_double>();

; __sycl_getScalar2020SpecConstantValue(@SymbolicID, %DefaultValue, i8 *%RTBuffer)
; is being replaced with

%gep = getelementptr i8, i8* %RTBuffer, i32 [calculated-offset-for-@SymbolicID]
%bitcast = bitcase i8* %gep to double*
%load = load double, double* %bitcast

; %load is the resulting value, which is used further instead of a result of
; call to __sycl_getScalar2020SpecConstantValue
```

The layout of that buffer is defined as follows: all specialization constants
are placed there one after another in ascending order of their numeric IDs
assigned to them by `SpecConstantPass` previously.

For example, the following code:
```
struct Nested {
  float a, b;
};
struct A {
  int x;
  Nested n;
};

constexpr specialization_id<int> id_int;
constexpr specialization_id<A> id_A;
constexpr specialization_id<Nested> id_Nested;
// ...
  [=](kernel_handler h) {
    h.get_specialization_constant<id_int>();
    h.get_specialization_constant<id_A>();
    h.get_specialization_constant<id_Nested>();
  }
```

Will result in the following buffer layout, i.e. offsets of each specialization
constant in that buffer:
```
[
  0, // for id_int, the first constant is at the beginning of the buffer
  4, // sizeof(int) == 4, the second constant is located right after the fisrt one
  16, // sizeof(int) + sizezof(A) == 4, the same approach for the third constant
]
```

#### 3. Collecting spec constants info and communicating it to DPC++ RT

For each encountered specialization constants `sycl-post-link` emits a property,
which encodes information required by DPC++ RT to set the value of a
specialization constant through corresponding API.

These properties are stored in "SYCL/specialization constants" property set and
their names are the same as symbolic IDs of corresponding specialization
constants (i.e. strings returned by `__builtin_sycl_unique_id` for associated
`specialization_id` variables).

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

constexpr specialization_id<int> id_int;
constexpr specialization_id<A> id_A;
// ...
  [=](kernel_handler h) {
    h.get_specialization_constant<id_int>();
    h.get_specialization_constant<id_A>();
  }
```

Will result in the following property set generated:
```
property_set {
  Name = "SYCL/specialization constants",
  properties: [
    property {
      Name: "id_int_symbolic_ID",
      ValAddr: points to byte array [{0, 0, 4}],
      Type: SYCL_PROPERTY_TYPE_BYTE_ARRAY,
      Size: sizeof(byte array above)
    },
    property {
      Name: "id_A_symbolic_ID",
      ValAddr: points to byte array [{1, 0, 4}, {2, 4, 4}, {3, 8, 4}],
      Type: SYCL_PROPERTY_TYPE_BYTE_ARRAY,
      Size: sizeof(byte array above)
    },
  ]
}
```

The property set described above is mainly intended to be used when native
specialization constants are available, but it will be also used for emulation
of specialization constants: SPIR-V IDs and sizes of specialization constants
will be used to calculate offset of each specialization constant within a
buffer, which is used to propagate them to kernel through kernel arguments.

Additionally, another property set will be generated to support emulated
specialization constants, which will contain a single property with default
values of all specialization constants in the same form as they will be
propagated from host to device through kernel arguments, i.e. this property will
simply contain a blob that for each specialization constant of type `A`
represents an object of type `A` constructed with values passed to
`specialization_id` constructor; those values are ordered in ascending order of
numeric SPIR-V IDs assigned to corresponding specialization constants.

This blob can be used by DPC++ RT to either pre-initialize the whole buffer for
specialization constants with their default value or to extract default value of
a particular specialization constant out of it.

For example, the following code:
```
struct Nested {
  constexpr Nested(float a, float b) : a(a + 1.0), b(b + 1.0) {}
  float a, b;
};
struct A {
  constexpr A(int x, float a, float b) : x(x), n(a, b) {}
  int x;
  Nested n;
};

constexpr specialization_id<int> id_int(42);
constexpr specialization_id<A> id_A(1, 2.0, 3.0);
constexpr specialization_id<Nested> id_Nested(4.0, 5.0);
// ...
  [=](kernel_handler h) {
    h.get_specialization_constant<id_int>();
    h.get_specialization_constant<id_A>();
    h.get_specialization_constant<id_Nested>();
  }
```

The following property set will be generated:
```
property_set {
  Name = "SYCL/specialization constants default values",
  properties: [
    property {
      Name: "all",
      ValAddr: points to byte array [
        42, // id_int
        1, 3.0, 4.0, // id_A
        5.0, 6.0 // id_Nested
      ],
      Type: SYCL_PROPERTY_TYPE_BYTE_ARRAY,
      Size: sizeof(byte array above)
    }
  ]
}
```

### DPC++ Compiler: front-end

DPC++ FE is responsible for several things related to specialization constants:

While transforming SYCL kernel function into an OpenCL kernel, DPC++ FE should
- Handle `kernel_handler` argument: it is not captured by lambda and therefore
  should be separately handled in DPC++ FE
- Communicate to DPC++ RT which kernel argument should be used for passing
  a buffer with specialization constant values when they are emulated.

DPC++ FE provides implementation of `__builtin_sycl_unique_id` built-in function and
it also populates special integration footer with the content required by DPC++
RT for access to right device image properties describing specialization
constants.

#### SYCL Kernel function transformations

`kernel_handler` is defined by SYCL 2020 specification as interface for
retrieving specialization constant values in SYCL kernel functions, but it
actually used only in emulation mode: since native specialization constant are
directly lowered into corresponding SPIR-V instructions, no additional handling
is needed. However, in order to get a value of a specialization constant which
was passed through a buffer, we need to have a pointer to that buffer: as it is
shown in DPC++ Headers section of the document, pointer to that buffer is stored
within `kernel_handler` object and passed to `__sycl_get*2020SpecConstantValue`
function.

According to the [compiler design][compiler-and-runtime-design], DPC++ FE wraps
SYCL kernel functions into OpenCL kernels and when `kernel_handler` object is
passed as an argument to SYCL kernel function, DPC++ FE should re-create that
object within the wrapper function and initialize it from implicitly created
OpenCL kernel argument.

Note: that extra `kernel_handler` object is not needed in every case: for,
example it is effectively unused when native specialization constants are
supported. However, per our [compiler-design][compiler-and-runtime-design], we
don't have per-target information about kernel signatures, which means that
kernel signatures must be the same for all targets, i.e. the same between
native and emulated specialization constants (JIT vs AOT compilation).

[compiler-and-runtime-design]: https://github.com/intel/llvm/blob/sycl/sycl/doc/CompilerAndRuntimeDesign.md#lowering-of-lambda-function-objects-and-named-function-objects

Considering the following input to DPC++ FE:
```
template <typename KernelName, typename KernelType>
  __attribute__((sycl_kernel)) void
  kernel_single_task(const KernelType &KernelFunc, kernel_handler kh) {
    KernelFunc(kh);
  }
```

It should be transformed into something like this:
```
__kernel void oclKernel(args_for_lambda_init, ..., specialization_constants_buffer) {
  KernelType LocalLambdaClone = { args_for_lambda_init }; // We already do this
  kernel_handler LocalKernelHandler;
  LocalKernelHandler.__init_specialization_constants_buffer(specialization_constants_buffer);
  // for simplicity we could have just used
  // kernel_handler LocalKernelHandler = { args_for_kernel_handler_init };
  // here, but we assume that kernel_handler might be used for more than just
  // accessing specialization constants and therefore there could be other
  // initialization parameters which also could be conditional
  // Even now we don't need to always initialize the kernel_handler object
  // Re-used body of "sycl_kernel" function:
  {
     LocalLambdaClone(LocalKernelHandler);
   }
}
```

The argument is communicated to DPC++ through regular integration header
mechanism, i.e. it is added as new entry to `kernel_signatures` structure there
with parameter kind set to a new enumeration value
`kernel_param_kind_t::kind_specialization_constants_buffer`.

#### `__builtin_sycl_unique_id`

This built-in is used to generate unique identifiers for specialization
constants, which are used in communication between the compiler and the runtime.

`__builtin_sycl_unique_id` is defined as follows: it accepts a variable and
returns a C-string (`const char *`), which:

- if the input variable has external linkage, the string must be the same in all
  translation units that pass this same variable to the built-in.
- if the input variable has internal linkage, the string must be unique across
  all translation units.
- return string must be the same if the built-in was called twice for the same
  variable within a single translation unit (regardless of its linkage type).

#### Integration footer generation

Note: we could have used `__builtin_sycl_unique_id` directly in DPC++ Headers,
but this would break compilation of those with a third-party C++ 17-compatible
compiler, which is unaware of this built-in function. Therefore, the compiler
generates a header file, which includes _the result_ of calling
`__builtin_sycl_unique_id` function and it is included into the user's program.
By doing so we can still use this non-standard built-in function and preserve
support for third-party host compilers.

However, as noted above, we can't use regular integration header here, because
in general case, `specialization_id` variables can't be forward-declared.
Therefore, we are using _integration footer_ approach, i.e. we generate a header
file which must be included at the end of a translation unit.

For the following code snippet:
```
struct A {
  float a, b;
};

constexpr specialization_id<int> id_int;
struct Wraper {
public:
  static constexpr specialization_id<A> id_A;
};
constexpr inline specialization_id<double> id_double;
constexpr inline specialization_id<float> id_float;
// ...
[&](handler &cgh) {
  cgh.set_specialization_constant<id_int>(42);
  cgh.get_specialization_constant<id_double>();
  // ...
  [=](kernel_handler h) {
    h.get_specialization_constant<id_int>();
    h.get_specialization_constant<Wrapper::id_A>();
  }
}
```

The integration footer will look like:

```
namespace sycl {
inline namespace _V1 {
namespace detail {
// Note: we do not declare `get_spec_constant_symbolic_ID` here and assume that
// it is declared in some other header which was already included.

// specializations for each specialization constant (for each `specialization_id`):
// we can refer to all those specialization_id variables, because integration
// footer was _appended_ to the user-provided translation unit
template<>
inline const char *get_spec_constant_symbolic_ID<id_int>() {
  return "result of __builtin_sycl_unique_id(id_int) encoded here";
}

template<>
inline const char *get_spec_constant_symbolic_ID<Wrapper::id_A>() {
  return "result of __builtin_sycl_unique_id(Wrapper::id_A) encoded here";
}

template<>
inline const char *get_spec_constant_symbolic_ID<id_double>() {
  return "result of __builtin_sycl_unique_id(id_double) encoded here";
}

template<>
inline const char *get_spec_constant_symbolic_ID<id_float>() {
  return "result of __builtin_sycl_unique_id(id_float) encoded here";
}

} // namespace detail
} // namespace _V1
} // namespace sycl

// Detailed description of this header is provided below in corresponding RT section
#include <sycl/detail/spec_const_integration.hpp>
```

Note that `get_spec_constant_symbolic_ID` specialization are generated for each
definition of `specialization_id` object regardless of its uses within SYCL
kernel functions: those IDs are used by DPC++ RT as well even for those spec
constants, which are never accessed on device.

##### Ambiguous references to specialization_id

There are valid C++ code examples, where references to `specialization_id`
variables could be ambiguous if they just referenced from a global namespace
like shown above. For example:

```
constexpr sycl::specialization_id<int> same_name{1};

/* application code that references "::same_name" */

namespace {
  constexpr sycl::specialization_id<int> same_name{2}:
  /* application code that referenes ::(unnamed)::same_name */
  namespace {
    constexpr sycl::specialization_id<int> same_name{3}:
    /* application code that referenes ::(unnamed)::(unnamed)::same_name */
  }
}

/* application code that references "::same_name" */
```

In that case we can't use `same_name` for specializing
`get_spec_constant_symbolic_ID`, because it would be ambiguous reference.
However, we can do the following trick:

```
// Content of integration footer for the example above

// For unambiguous references we can generate regular specialization
template<>
inline const char *get_spec_constant_symbolic_ID<::same_name>() {
  return "result of __builtin_sycl_unique_id(::same_name) encoded here";
}

// For ambiguous references we generate 'shim' functions, which allows us to
// get an address of a variable within a (possible nested) anonymous namespace
// without spelling it.
namespace {
  namespace __sycl_detail {
    // This helper is need to get addresses of variables defined within
    // anonymous namespace.
    // It is generated for each specialization_id within an anonymous namespace
    // if there is the same specialization_id defined in global namespace
    static constexpr decltype(same_name) &__shim_0() {
      // address of ::(unnamed)::same_name;
      return same_name;
    }
  }
}
namespace sycl {
  namespace detail {
    // By using 'shim' function were are able to unambiguously refer to a
    // variable within an anonymous namespace
    template<>
    inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__shim_0()>() {
      return "unique id for ::(unnamed)::same_name";
    }
  }
}
namespace {
  namespace {
    namespace __sycl_detail {
      static constexpr decltype(same_name) &__shim_1() {
        // address of ::(unnamed)::(unnamed)::same_name;
        return same_name;
      }
    }
  }

  namespace __sycl_detail {
    // Sometimes we need a 'shim', which points to another 'shim' in order to
    // "extract" a variable from an anonymous namespace unambiguosly
    static constexpr decltype(__sycl_detail::__shim_1()) &__shim_2() {
      // still address of ::(unnamed)::(unnamed)::same_name;
      return __sycl_detail::__shim_1();
    }
  }
}
namespace sycl {
  namespace detail {
    template<>
    inline const char *get_spec_constant_symbolic_ID<::__sycl_detail::__shim_2()>() {
      return "unique id for ::(unnamed)::(unnamed)::same_name";
    }
  }
}

#include <sycl/detail/spec_const_integration.hpp>
```

### DPC++ runtime

For each device binary compiler generates a map
\<Symbolic ID\> =\> \<list of spec constant descriptors\> ("ID map"). DPC++
runtime imports that map when loading device binaries.
It also maintains another map \<Symbolic ID\> =\> \<its value\> ("value map")
per `sycl::kernel_bundle` object. The value map is updated upon
`kernel_bundler::set_specialization_constant<ID>(val)` and
`handler::set_specialization_constant<ID>(val)` calls from the app.

In order for runtime to access the right property, it need to compute the
symbolic ID of a specialization constant based on user-provided inputs, such
as non-type template argument passed to `set_specialization_constant` argument.
DPC++ Headers section describes how symbolic IDs are generated and the same
trick is used within `set_specialization_constant` method:
```
template<auto& SpecName>
void set_specialization_constant(
  typename std::remove_reference_t<decltype(SpecName)>::type value) {
  const char *SymbolicID = detail::get_spec_constant_symbolic_ID<SpecName>();
  // remember the value of the specialization constant
  SpecConstantValuesMap[SymbolicID] = value;
}
```

Before invoking JIT compilation of a program, the runtime "flushes"
specialization constants:

If native specialization constants are supported by the target device, the
runtime iterates through the value map and invokes

```
ur_result_t urProgramSetSpecializationConstants(ur_program_handle_t hProgram,
                                                uint32_t count,
                                                const ur_specialization_constant_info_t *
                                                pSpecConstants)
```

Unified Runtime function for descriptor of each property. The definition for
`ur_specialization_constant_info_t` is as follows:

```
struct ur_specialization_constant_info_t {
    uint32_t id;
    size_t size;
    const void *pValue;
};
```

`id` and `size` are taken from the descriptor, `pValue` is calculated based on
address of the specialization constant provided by user and `offset` field of
the descriptor as `(char*)(SpecConstantValuesMap[SymbolicID]) + offset`.

That calculation is required, because at SPIR-V level composite
specialization constants are respresented by several specialization constants
for each element of a composite, whilst on a SYCL level, the whole composite
is passed by user as a single blob of data. `offset` field from properties is
used to specify which exact piece of that blob should be extracted to perform
per-element composite specialization constant initialization.

If native specialization constants are not supported by the target device, then
the runtime calculates the location (offset) of each specialization constant in
corresponding runtime buffer and copied user-provided value into that location.

That buffer should be allocated for each `device_image` and
it should be set as a kernel argument, if corresponding `kernel_signature`
contains `kernel_param_kind_t::kind_specialization_constants_buffer`.

Offsets into that buffer are calculated based on "SYCL/specialization constants"
property set, i.e. all properties from there are sorted in ascending order of
their numeric IDs and offset for each specialization constant is calculated as
sum of sizes of all other specialization constants with smaller numeric ID.

In order to properly set default values of specialization constants,
"SYCL/specialization constants default values" property set is used: its content
is used to either fully or partially initialize the buffer with specialization
constant values.

#### sycl/detail/spec_const_integration.hpp header file

DPC++ RT needs to have access to a mapping between `specialization_id` variables
and corresponding unique symbolic IDs used by the compiler. As already stated
above, we use integration footer for that by providing template specializations
of `get_spec_constant_symbolic_ID` function template.

The tricky thing here, is that C++ specification states the following:

> Specialization must be declared before the first use that would cause implicit
> instantiation, in every translation unit where such use occurs.
>
> [cppreference][cppreference-template-specialization]

[cppreference-template-specialization]: https://en.cppreference.com/w/cpp/language/template_specialization

That means that all users of `get_spec_constant_symbolic_ID` has to appear
*after* we defined all `get_spec_constant_symbolic_ID` template specializations.

`sycl/detail/spec_const/integration.hpp` header file is intended to be a
location for such methods/classes/functions.

### SPIRV-LLVM-Translator

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
              OpDecorate %A.i32 SpecId 43                          ; ID of the 1st member
              OpDecorate %A.float SpecId 44                        ; ID of the 2nd member
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
