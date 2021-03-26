# Specialization constants

Specialization constants are implemented in accordance with how they are defined
by SYCL 2020 specification: [SYCL registry][sycl-registry],
[direct link to the specification][sycl-2020-spec].

[sycl-registry]: https://www.khronos.org/registry/SYCL/
[sycl-2020-spec]: https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html

TODO: feature overview? code example?

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

4. When AOT compilation is used or target is a CUDA device (where NVPTX
intermediate representation is used), we need to somehow emulate support for
specialization constants.

## Design

As stated above, native specialization constants support is based on
corresponding SPIR-V functionality, while emulation is supposed to be
implemented through transforming specialization constants into kernel arguments.

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
(numeric IDs in SPIR-V or kernel arguments).

That mapping mechanism is particularly tricky, because of some additional
complexity coming from SYCL 2020 specification:
- `specialization_id` variables, which are used as specialization constant
  identifiers (being non-type template parameters of some methods) can't be
  forward-declared in general case (for example, if defined as `static`), which
  means that we can't use integration header to attach some information to them
  through some C++ templates tricks (like it is done for kernel arguments or
  kernel names, for example).
- they also can be declared as `static` or just non-`inline` `constexpr`, which
  means that they have internal linkage and can't be referenced from other
  translation units, which means that we can't for example create a new
  translation unit which contains some mapping from `specialization_id` address
  to some desired info.

Based on those limitations, the following design is proposed:
- DPC++ RT uses special function:
  ```
  namespace detail {
    template<auto &SpecName>
    const char *get_spec_constant_symbolic_ID();
  }
  ```
  Which is only declared, but not defined in there and used to retrieve required
  information like numeric ID of a specialization constant.
- Definition of that function template are provided by DPC++ FE in form of
  _integration footer_: the compiler generates a piece of C++ code which is
  injected at the end of the translation unit:
  ```
  namespace detail {
    // assuming user defined and used the following specialization_id:
    // constexpr specialiation_id<int> int_const;
    // class Wrapper {
    // public:
    //   static constexpr specialization_id<float> float_const;
    // };

    template<>
    const char *get_spec_constant_symbolic_ID<int_const>() {
      return "unique_name_for_int_const";
    }
    template<>
    const char *get_spec_constant_symbolic_ID<Wrapper::float_const>() {
      return "unique_name_for_Wrapper_float_const";
    }
  }
  ```

  Those symbolic IDs are used to identify device image properties corresponding
  to those specialization constants, which store additional information (like
  numeric SPIR-V ID of a constant) needed for DPC++ RT.
- That integration footer is automatically embedded by the compiler at the end
  of user-provided translation unit by driver.

Summarizing, overall design looks like:

DPC++ Headers provide special markup, which used by the compiler to detect
presence of specialization constants and properly handle them.

DPC++ FE handles `kernel_handler` SYCL kernel function argument, creates
additional kernel arguments to pass specialization constants through buffer if
necessary (if native support is not available) and generates integration footer.

`sycl-post-link` transforms device code to either generate proper SPIR-V with
specialization constants (when native support is available) or to generate
correct access to corresponding kernel arguments (which are used when native
support is not available); also the tool generates some device image properties
with all information needed for DPC++ RT (like which numeric SPIR-V ID was
assigned to which symbolic ID).

With help of `clang-offload-wrapper` tool, those device image properties are
embedded into the application together with device code and used by DPC++ RT
while handling specialization constants during application execution: it either
calls corresponding PI API to set a value of corresponding specialization
constant or it fills a special buffer with values of specialization constants
and passes it as kernel argument to emulate support of specialization constants.

Sections below describe each component in details.

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
// TODO: Add `const` to `DefaultValue` and `RTBuffer`?
template<typename T>
T __sycl_getScalar2020SpecConstantValue<T>(const char *SymbolicID, void *DefaultValue, void *RTBuffer);
template<typename T>
T __sycl_getComposite2020SpecConstantValue<T>(const char *SymbolicID, void *DefaultValue, void *RTBuffer);

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
    auto ID = __builtin_unqiue_ID(SpecName);
    return __sycl_getScalar2020SpecConstantValue<T>(ID, &S, Ptr);
  }

  template<auto &SpecName, typename T = std::remove_reference_t<decltype(SpecName)>::type>
  // enable_if T is a composite type
  T get_on_device() {
    auto ID = __builtin_unqiue_ID(SpecName);
    return __sycl_getComposite2020SpecConstantValue<T>(ID, &S, Ptr);
  }
#endif // __SYCL_DEVICE_ONLY__

  byte *Ptr = nullptr;
};

} // namespace sycl
```

Here `__builtin_unique_ID` is a new compiler built-in which is supposed to
generate unique symbolic IDs for specialization constants.

`__sycl_getScalar2020SpecConstantValue<T>` and
`__sycl_getComposite2020SpecConstant<T>` are functions with special names - they
are declared in the headers but never defined. Calls to them are recognized by
a special LLVM pass later and this is aforementioned special markup required for
the compiler.

Those functions accept three parameters:
1. Symbolic ID of specialization constant.
   TODO: do we need more details here?

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
2. Brings IR to the form expected by the SPIR-V translator (format of the
   expected IR is covered in "Transformation of LLVM IR to SPIR-V friendly IR
   form" section)
3. Collects and provides \<Symbolic ID\> =\> \<numeric IDs + additional info\>
   mapping, which is later being used by DPC++ RT to set specialization constant
   values provided by user (section "Collecting spec constants info and
   communicating it to DPC++ RT" provides more info on that)

#### Assignment of numeric IDs to specialization constants

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

specialization_id<int> id_int;
specialization_id<A> id_A;
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

##### Transformation of LLVM IR to SPIR-V friendly IR form

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
`T __sycl_getScalar2020SpecConstantValue(const char *SymbolicID, void *DefaultValue, char *RTBuffer)`
intrinsic into calls to `T __spirv_SpecConstant(int ID, T default_val)`.
And for `T __sycl_getComposite2020SpecConstantValue(const chat *SybmolicID, void *DefaultValue, char *RTBuffer)`
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

specialization_id<int> id_int;
specialization_id<A> id_A;
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
      Type: PI_PROPERTY_TYPE_BYTE_ARRAY,
      Size: sizeof(byte array above)
    },
    property {
      Name: "id_A_symbolic_ID",
      ValAddr: points to byte array [{1, 0, 4}, {2, 4, 4}, {3, 8, 4}],
      Type: PI_PROPERTY_TYPE_BYTE_ARRAY,
      Size: sizeof(byte array above)
    },
  ]
}
```

#### DPC++ runtime

For each device binary compiler generates a map
\<Symbolic ID\> =\> \<list of spec constant descriptors\> ("ID map"). DPC++
runtime imports that map when loading device binaries.
It also maintains another map \<Spec const symbolic ID\> =\> \<its value\> 
("value map") per `sycl::kernel_bundle` object. The value map is updated upon
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
  const char *SymbolicID =
#if __has_builint(__builtin_unique_stable_name)
      __builtin_unique_stable_name(detail::specialization_id_name_generator<S>);
#else
      // without the builtin we can't get the symbolic ID of the constant
      "";
#endif
  // remember the value of the specialization constant
  SpecConstantValuesMap[SymbolicID] = value;
}
```

The major downside of that approach is that it can't be used with any
third-party host compiler, because it uses a specific built-in function to
generate symbolic IDs of specialization constants. Good solution would be to
employ integration header here, i.e. we could provide some class template
specializations which will return symbolic IDs - the same approach as we use
for communicating OpenCL kernel names from the compiler to the runtime.

For the following user code:
```
specalization_id<int> id_int;
// ...
  [=](kernel_handler h) {
    h.get_specialization_constant<id_int>();
  }
```

The following integration header would be produced:
```
// fallback
template<auto &S>
class specialization_constant_info {
  static const char *getName() { return ""; }
};

// forward declaration
extern specialization_id<int> id_int;

// specialization
template<>
class specialization_constant_info<id_int> {
  static const char *getName() {
    return "result of __builtin_unique_stable_name(detail::specialization_id_name_generator<id_int>) encoded here";
  }
};
```

And it would be used by DPC++ RT in the following way:
```
template<auto& SpecName>
void set_specialization_constant(
  typename std::remove_reference_t<decltype(SpecName)>::type value) {
  const char *SymbolicID = specialiation_constant_info<SpecName>::getName();
  // remember the value of the specialization constant
  SpecConstantValuesMap[SymbolicID] = value;
}
```

Such trick would allow use to compile host part of the app with any third-party
compiler that supports C++17, but the problem here is that SYCL 2020 spec states
the following:

> Specialization constants must be declared using the `specialization_id` class,
> and the declaration must be outside of kernel scope using static storage
> duration. The declaration must be in either namespace scope or class scope.

`class` scope `static` variables are not forward-declarable, which means that
the approach with integration header is not available for us here.

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

### Emulation of specialization constants

Emulation of specialization constants is performed by converting them into
kernel arguments.

Overall idea is that DPC++ runtimes packs all specialization constants into a
single buffer, which is passed as an extra implicit kernel argument. Then the
compiler instead of lowering `__sycl_get*2020SpecConstantValue` intrinsics into
SPIR-V friendly IR replaces it with extracting an element from that buffer.

"All" specialization constants here means complete list of specialization
constants encountered in an application or a shared library which is being
compiled: that list is computed by `sycl-post-link` tool and communicated to
the runtime through device image properties like it is described in "Support for
native specialization constants" section.

#### DPC++ Headers

The same DPC++ Headers are used for native and emulated specialization constants
and their design is decribed in the corresponding sub-section of "Support for
native specialization constants" section.

However, that part of the document doesn't describe the third argument of
`__sycl_get*2020SpecConstantValue` intrinsics: it is a pointer to a runtime
buffer, which holds values of all specialization constants and should be used
to retrieve their values in device code.

This pointer is stored within `kernel_handler` object and it is initialized only
if our target doesn't support native specialization constants.
Since `kernel_handler` object is not captured by SYCL kernel funtion, it means
that we are not able to employ some header-only solution here and need help of
the compiler.

DPC++ FE searches for functions marked with `sycl_kernel` attribute to handle
them and turn into entry points of device code.


#### DPC++ FE

DPC++ FE should look for `kernel_handler` argument in a function marked with
`sycl_kernel" attribute. If such argument is present, it means that this kernel
can access specialization constants and therefore FE needs to do the following:

If native specialization constants are supported:
- create `kernel_handler` object
- use default constructor to initialize it
- pass that `kernel_handler` object to user-provided SYCL kernel function

If native specialization constants are not supported:
- generate one more kernel argument for passing a buffer with specialization
  constants values
- create `kernel_handler` object
- initialize that `kernel_handler` object with newly created kernel argument
- pass that `kernel_handler` object to user-provided SYCL kernel function
- Provide information about new kernel argument through the integration header

So, having the following as the input:
```
template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void
kernel_single_task(const KernelType &KernelFunc, kernel_handler kh) {
  KernelFunc(kh);
}
```
For the target which doesn't have native support for specialization constatns
DPC++ FE shoud tranform it into something like:

```
__kernel void KernelName(args_for_lambda_init, ..., char *specialization_constants_buffer) {
  KernelType LocalLambdaClone = { args_for_lambda_init }; // We already do this
  kernel_handler LocalKernelHandler;
  LocalKernelHandler.__init_specialization_constants_buffer(specialization_constants_buffer);
  // Re-used body of "sycl_kernel" function:
  {
     LocalLambdaClone(LocalKernelHandler);
  }
}
```

Also the new kernel argument `specialization_constants_buffer` should have
corresponding entry in the `kernel_signatures` structure in the integration
header. The param kind for this argument should be
`kernel_param_kind_t:specialization_constants_buffer`.

Example:
```
  const kernel_param_desc_t kernel_signatures[] = {
   //--- _ZTSN2cl4sycl6detail19__pf_kernel_wrapperIZZ4mainENK3$_0clERNS0_7handlerEE6init_aEE
   { kernel_param_kind_t::kind_std_layout, 8, 0 },
   { kernel_param_kind_t::kind_accessor, 4062, 8 },
   { kernel_param_kind_t::kind_specialization_constants_buffer, /*parameter_size_in_bytes= */ 8, /*offset_in_lambda=*/0},
 };

```

Offset for this argument is zero since it has no any connected captured
variable.

#### DPC++ Compiler: sycl-post-link tool

When native specialization constants are not available, we need to lower
`__sycl_get*2020SpecializationConstant` intrinsic into some load from the
additional kernel argument, which points to a buffer with all specialization
constant values.

We assume that both DPC++ compiler and runtime know the layout of that buffer so
the compiler can correctly access particular constants from it and the runtime
is able to properly fill the buffer with values of those specialization
constants.

The layout is defined as follows: all specialization constants are sorted by
their numeric IDs (i.e. the order of they discovery by sycl-post-link tool) and
stored within a buffer one after each other without any paddings. So, the
specialization constant with ID `N` is located within a buffer at offset, which
is equal to sum of sizes of all specialization constants with ID less than `N`.

For example, if we have the following specialization constants discovered in the
following order:
```
struct custom_type { int a; double b; }
specialization_id<double> id_double;
specialization_id<custom_type> id_custom;
specialization_id<int> id_int;
```
`id_double` will be located at the beginning of the buffer, because it is the
first discovered specialization constant (ID = 0). `id_custom` (ID = 1) will be
located at the offset 8, because we have a single specialization constant with
the ID < 1 and its size is 8 bytes. `id_int` (ID = 2) will be located at the
offset 20, which is computed as `sizeof(id_double) + sizeof(id_custom)`.

When specialization constants emulation is requested, `sycl-post-link` replaces
calls to `__sycl_get*SpecializationConstant` intrinsics with the following
LLVM IR pattern:
```
%gep = i8, i8* %arg_three_of_sycl_intrinsic_call, i64 [offset]
; We use the third argument of the __sycl_get*SpecializationConstant intrinsic
; as a pointer to where all specialization constants are stored
; [offset] here is a placeholder for some literal integer value computed by
; the pass based on the ID of the requested specialization constant as described
; above
%cast = bitcast i8* %gep to [return-type]*
; [return-type] here is a placeholder for the actual type of the requested
; specialization constant
%load = load [return-type], [return-type]* %cast
; %load is the resulting value, which should replace all uses of the original
; call to __sycl_get*SpecializationConstant intrinsic
```

**TODO**: elaborate on handling of composite types.

##### Collecting spec constants info and communicating it to DPC++ RT

As in the processing of native specialization constants, `sycl-post-link` emits
some information in device image properties, which is required by DPC++ runtime
to properly handle emulation of specialization constants.

`sycl-post-link` provides two property sets when specialization constants are
emulated:
1. Mapping from Symbolic ID to offset
2. Mapping from Symbolic ID to the default value

The first mapping can be substituted with the property set generated for native
specialization constants, but it is still provided in order to simplify the
runtime part, i.e. it allows to avoid calculating those offsets at runtime by
re-using ones calculated by the compiler.

**TODO**: is it possible to have both native and emulated specialization
constants within a single device image?

The second mapping is required and it allows the runtime to properly set default
values of specialization constants.

**TODO**: document exact property set names and properties structure

#### DPC++ Compiler: Generation of OpenCL kernel

Optional `kernel_handler` SYCL kernel function argument should be created by
front-end and passed to SYCL kernel function if it is expected there.

So, the following SYCL code
```
specialization_id<int> id_int;
class WithSpecConst;
class WithoutSpecConst;
// ...
/* ... */.single_task<WithSpecConst>([=](kernel_handler h) {
  auto v = h.get_specialization_constant<id_int>();
  // ...
});
/* ... */.single_task<WithoutSpecConst>([=]() {
  // ...
});
```

Should produce something like this (pseudo-code):
```
void WithSpecConstOpenCLKernel(/* ... */) {
  kernel_handler h;
  WithSpecConstSYCLKernelFunction(/* ... */, h);
}
void WithoutSpecConstOpenCLKernel(/* ... */) {
  WithoutSpecConstSYCLKernelFunction(/* ... */);
}
```
