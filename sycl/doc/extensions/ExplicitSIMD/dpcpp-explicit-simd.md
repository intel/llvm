# Explicit SIMD Programming Extension for DPC++

## Introduction

The main motivation for introducing the Explicit SIMD Programming (ESP) DPC++
extension is enabling low-level efficient programming for Intel Gen
architectures. More specifically, Explicit SIMD provides the following
additional features:
- Manual vectorization of device code using the `simd` class mapped to Gen's
  general register file. This allows to write efficient code not relying on
  further widening by the compiler, as with traditional SPMD programming.
- Low-level APIs efficiently mapped to Gen architecture, such as block reads.

Here are some future directions in which this API is intended to evolve:
- enabling this extension for other architectures, such as x86, with extracting
  and clearly marking generic and target-dependent API portions
- aligning with `std::simd` and maybe providing `std::simd` implemenation atop
  `sycl::intel::gpu`

## Explicit SIMD execution model

Explicit SIMD execution model is a basically an equivalent of the base SYCL
execution model with subgroup size restricted to 1. Which means each subgroup
maps to a single hardware thread. All standard SYCL APIs continue to work,
including `sycl::intel::sub_group` ones, which become either a no-op or
trivial. E.g. a barrier becomes just a memory fence for a compiler, collectives
just return the value in the single work-item. Another consequences of the unit
subgroup size is guaranteed independent forward progress between work-items on
some Gen architecture generations.

## Explicit SIMD extension APIs

Explicit SIMD APIs can be used only in code to be executed on Intel Gen
architecture devices and the host device for now. Attempt to run such code on
other devices will result in error. 

Kernels and `SYCL_EXTERNAL` functions using ESP must be explicitly marked with
the `[[sycl_explicit_simd]]` attribute. Subgroup size query within such
functions will always return `1`.

*Functor kernel*
```cpp
using AccTy = cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                                 cl::sycl::access::target::global_buffer>;
class Functor {
public:
  Functor(int X, AccTy &Acc) : X(X), Acc(Acc) {}

  [[sycl_explicit_simd]] void operator()() { Acc[0] += X; }

private:
  int X;
  AccTy Acc;
};
```

*Lambda kernel and function*
```cpp
[[sycl_explicit_simd]] SYCL_EXTERNAL
void sycl_device_f(cl::sycl::global_ptr<int> ptr, sycl::intel::gpu::simd<float, 8> X) {
  sycl::intel::gpu::flat_block_write(*ptr.get(), X);
}
...
  Q.submit([&](cl::sycl::handler &Cgh) {
    auto Acc1 = Buf1.get_access<cl::sycl::access::mode::read>(Cgh);
    auto Acc2 = Buf2.get_access<cl::sycl::access::mode::read_write>(Cgh);

    Cgh.single_task<class KernelID>([=] () [[sycl_explicit_simd]] {
      sycl::intel::gpu::simd<float, 8> Val = sycl::intel::gpu::flat_block_read(Acc1.get_pointer());
      sycl_device_f(Acc2, Val);
    });
  });
}
```

## Implementation restrictions
Current ESP implementation does not support using certain standard SYCL features
inside explicit SIMD kernels and functions. Most of them will be eventually
dropped. What's not supported today:
- Mixing `[[sycl_explicit_simd]]` kernels with SYCL kernels in a single source
- Local accessors. Local memory is allocated and accessed via explicit
device-side API
- 2D and 3D accessors
- Constant accessors
- `sycl::accessor::get_pointer()`. All memory accesses through an accessor are
done via explicit APIs; e.g. `sycl::intel::gpu::block_store(acc, offset)`
- Few others (to be documented)


## Core Explicit SIMD programming APIs

The DPC++ Explicit SIMD library defines the following classes to enhance the
expressiveness for explicit SIMD data-parallel programming while enabling
efficient mapping to SIMD vector operations on Intel GPU architectures.

### SIMD vector class

The `sycl::intel::gpu::simd` class is a vector templated on some element type. The element type must be vectorizable type. The set of vectorizable types is the set of fundamental SYCL arithmetic types (C++ arithmetic types or `half` type) excluding `bool`. The length of the vector is the second template parameter.

Each simd class object is mapped to a consecutive block of general register
files (GRF).

```cpp
namespace sycl {
namespace intel {
namespace gpu {

// __vector_type, using clang vector type extension.
template <typename __Ty, int __N> struct __vector_type {
  static_assert(!std::is_const<__Ty>::value, "const element type not supported");
  static_assert(__is_vectorizable_v<__Ty>::value, "element type not supported");
  static_assert(__N > 0, "zero-element vector not supported");

  static constexpr int length = __N;
  using type = __Ty __attribute__((ext_vector_type(__N)));
};

template <int __N>
using __mask_type_t = typename __vector_type<uint16_t, __N>::type;

template <typename __Ty, int __N> class simd {
public:
  using value_type = simd<__Ty, __N, 0>;
  using element_type = __Ty;
  using vector_type = __vector_type_t<__Ty, __N>;
  using type = simd<__Ty, __N>;
  static constexpr int length = __N;

  // Constructors.
  constexpr simd();
  constexpr simd(const simd &other);
  constexpr simd(simd &&other);
  constexpr simd(const vector_type &__Val);
  constexpr simd(std::initializer_list<__Ty> __Ilist) noexcept;
  constexpr simd(__Ty __Val, __Ty __Step = __Ty()) noexcept;

  // Assignment operators.
  constexpr type &operator=(const type &) &;
  constexpr type &operator=(type &&) &;

  // Subscript operators.
  __Ty operator[](int __i);
  __Ty operator[](int __i) const;

  // Unary operators.
  type &operator++();
  type operator++(int);
  type &operator--();
  type operator--(int);

  // Binary operators.
  auto operator +(const type &__RHS) const;
  auto operator -(const type &__RHS) const;
  auto operator *(const type &__RHS) const;
  auto operator /(const type &__RHS) const;
  type operator &(const type &__RHS) const;
  type operator |(const type &__RHS) const;
  type operator ^(const type &__RHS) const;

  // Compound assignment.
  type &operator +=(const type &__RHS);
  type &operator -=(const type &__RHS);
  type &operator *=(const type &__RHS);
  type &operator /=(const type &__RHS);
  type &operator &=(const type &__RHS);
  type &operator |=(const type &__RHS);
  type &operator ^=(const type &__RHS);

  // Compare operators.
  simd<uint16_t, __N> operator >(const type &__RHS) const;
  simd<uint16_t, __N> operator >=(const type &__RHS) const;
  simd<uint16_t, __N> operator <(const type &__RHS) const;
  simd<uint16_t, __N> operator <=(const type &__RHS) const;
  simd<uint16_t, __N> operator ==(const type &__RHS) const;
  simd<uint16_t, __N> operator !=(const type &__RHS) const;

  // Select operations.
  template <int __Size, int __Stride>
  simd<__Ty, __Size> select(uint16_t __Offset = 0) &&;
  template <int __Size, int __Stride>
  simd_view<type, region1d_t<__Ty, __Size, __Stride>>
  select(uint16_t __i = 0) &;

  // Replicate operations.
  template <int Rep> simd<__Ty, Rep * __N> replicate();
  template <int Rep, int W> simd<__Ty, Rep * W> replicate(uint16_t __i);
  template <int Rep, int VS, int W>
  simd<__Ty, Rep * W> replicate(uint16_t __i);
  template <int Rep, int VS, int W, int HS>
  simd<__Ty, Rep * W> replicate(uint16_t __i);

  // Format operations.
  template <typename __EltTy> auto format() &;
  template <typename __EltTy, int __Height, int __Width> auto format() &;

  // Merge operations.
  void merge(const value_type &__Val, const __mask_type_t<__N> &__Mask);
  void merge(const value_type &__Val1, value_type __Val2,
             const __mask_type_t<__N> &__Mask);
} // namespace gpu
} // namespace intel
} // namespace sycl
```

Every specialization of ```sycl::intel::gpu::simd``` shall be a complete type. The term
simd type refers to all supported specialization of the simd class template.
To access the i-th individual data element in a simd vector, Explicit SIMD supports the
standard subscript operator ```[]```, which returns by value.

For simd type object, Explicit SIMD supports the following simd vector operations:
- Unary operators: ++ (*pre-/post-increment*), -- (*pre-/post-decrement*)
- Binary operators: +, -, *, /, &, |, ^
- Compound assignments: +=, -=, *=, /=, &=, |=, ^=
- Compare operators: >, >=, <, <=, ==, !=

These are all element-wise operations, which apply a specified operation to the
elements of one or more simd objects and should follow the standard C++ rules for
the corresponding scalar data element computation. Each such application is
unsequenced with respect to the others.

To reference a subset of the elements in simd vector object, Explicit SIMD provides ```select```
function, which returns a `simd` or `simd_view` object (*described below*) representing
the selected sub-vector starting from the i-th element. The number of selected
elements is specified by the template parameter **__Size**, while the distance
between two adjacent elements is specified by the template parameter **__Stride**.

```cpp
  simd<int, 8> a;
  simd<int, 4> b;
  // ...
  b = a.select<4, 2>(1);  // size=4, stride=2, offset=1 (elements a(1),
                          // a(3), a(5) and a(7) are copied to b)
```

<p align="center">
<img src="images/VectorOdd.svg" title="1D select example" width="600" height="60"/>
</p>

```cpp
a.select<4, 2>(0) = b;  // selected elements of a are replaced
                        // with elements of b (all elements of b are
                        // copied to elements a(0), a(2), a(4), a(6))
```

<p align="center">
<img src="images/VectorEven.svg" title="1D select example" width="600" height="60"/>
</p>


Gen ISA provides powerful register region addressing modes to facilitate cross-lane
SIMD vector operation. To exploit this feature Explicit SIMD provides ```replicate``` function
to allow programmer to implement any native Gen ISA region in the following forms:
- ```replicate<REP>()```: replicate a simd vector object **REP** times and return a new simd
vector of **REP** * Width, where Width specifies the original vector size.
- ```replicate<REP, W>(uint16_t i)```: replicate **W** consecutive elements starting at the
i-th element from the simd vector object **REP** times, and return a new simd vector of **REP** * **W** length.
- ```replicate<REP, VS, W>(uint16_t i)```: replicate **REP** blocks of **W** consecutive
elements starting at the i-th from the simd vector object with each block strided by **VS**
elements, and return a new vector of **REP** * **W** length. Selected blocks of **W**
elements will overlap if **VS** < **W**.
- ```replicate<REP, VS, W, HS>(uint16_t i=0 )```: replicate **REP** blocks of **W** sequential
elements with a stride of **HS** starting at the i-th element from the simd vector object with
each block strided by **VS** elements, and return a new vector of **REP** * **W** length.
Selected blocks of **W** elements will overlap if **VS** < **W**.

To avoid explicit type cast and the resulting move instructions for large vectors, Explicit SIMD allows
programmer to reinterpret the fundamental data element type of a simd vector object and change
its shape to 1D or 2D object through the ```format``` function:
- ```format<__EltTy>( )```: returns a reference to the calling simd object interpreted as a new
simd vector with the size determined by the template **__EltTy** parameter.
- ```format<__EltTy, __Height, __Width>( )```: returns a reference to the calling simd object interpreted
as a new 2D simd_view object with the shape determined by the template parameters **__Height** and**__Width**. The size of the new 2D block must not exceed the size of the original object.

```cpp
  simd<int, 16> v1;
   // ...
  auto v2 = v1.format<short>;
  // v2 is a reference to the location of v1
  // interpreted as a vector of 32 shorts.
  // ...
  auto m1 = v1.format<int, 4, 4>;
  // m1 is a reference to the location of v1
  // interpreted as a matrix 4x4 of ints.
  // ...
  auto m2 = v1.format<char, 4, 16>( );
  // m2 is a reference to the location of v1
  // interpreted as a matrix 4x16 of chars.
```

To model predicated move, Explicit SIMD provides the following merge functions:
- ```merge(value_type __Val, __mask_type __Mask)```: this merge operation takes one source operand **__Val** and a mask **__Mask** defined as unsigned short vector of the same length. The semantic is that if the LSB of an element of **__Mask** is set, then the corresponding data element of **__Val** is copied to the corresponding position in the calling simd object.

```cpp
  simd<int, 4>   m, src;
  simd<unsigned short, 4> mask;
  // ...
  m.merge(src, mask);
  // m           src         mask      --->  m
  // 2 2 2 2     4 4 4 4     1 1 0 1         4 4 2 4
```

- ```merge(value_type __Val1, value_type __Val2, __mask_type __Mask)```: this merge operation takes two source operands **__Val1** and **__Val2** as well as a simd mask. The semantic is that if the LSB of an element of **__Mask** is set, then the corresponding data element of **__Val1** is copied to the corresponding position in the calling simd object. Otherwise the corresponding data element of **__Val2** is copied to the corresponding position in the calling simd object.

```cpp
  simd<int,4>   m, src1, src2;
  simd<unsigned short, 4> mask;
  // ...
  m.merge(src1, src2, mask);
  // m           src1        src2         mask        --->  m
  // 2 2 2 2     4 4 4 4     3 3 3 3      1 1 0 1           4 4 3 4
```
### `simd_view` class

The ```sycl::intel::gpu::simd_view``` represents a "window" into existing simd object,
through which a part of the original object can be read or modified. This is a
syntactic convenience feature to reduce verbosity when accessing sub-regions of
simd objects. **__RegionTy** describes the window shape and can be 1D or 2D,
**__BaseTy** is the original simd object type, which can be a ```simd_view```
itself.

```simd_view``` allows to model hierarchical "views" of the parent ```simd```
object's parts, read/modify its elements through the views. Views can be of
different shapes and dimensions as illustrated below (`auto` resolves to a
`simd_view` instantiation):

<p align="center">
<img src="images/simd_view.svg" title="1D select example" width="800" height="300"/>
</p>

```cpp
namespace sycl {
namespace intel {
namespace gpu {
template <typename __BaseTy, typename __RegionTy> class simd_view {
public:
  using __ShapeTy = typename __shape_type<__RegionTy>::type;
  static constexpr int length = __ShapeTy::__Size_x * __ShapeTy::__Size_y;
  using type = simd_view<__BaseTy, __RegionTy>;
  using value_type = simd<typename __ShapeTy::element_type, length>;
  using vector_type = __vector_type_t<typename __ShapeTy::element_type, length>;
  using region_type = __RegionTy;
  using base_type = typename __compute_base_type<__BaseTy, __RegionTy>::type;
  using element_type = typename __ShapeTy::element_type;

  // Constructors.
  simd_view(__BaseTy &__Base, __RegionTy __Region);
  simd_view(__BaseTy &&__Base, __RegionTy __Region);
  simd_view(type &__Other);
  simd_view(type &&__Other);

  // Assignment operators.
  type &operator=(const type &__Other);
  type &operator=(const value_type &__Val);

  // Region accessors.
  static constexpr bool is1D();
  static constexpr bool is2D();
  static constexpr int getSizeX();
  static constexpr int getStrideX();
  static constexpr int getSizeY();
  static constexpr int getStrideY();
  constexpr uint16_t getOffsetX();
  constexpr uint16_t getOffsetY();
  template <int __Dim = 0> static constexpr int getSize();
  template <int __Dim = 0> static constexpr int getStride();
  template <int __Dim = 0> constexpr uint16_t getOffset() const;

  // Subscript operators.
  element_type operator[](int __i);
  element_type operator[](int __i) const;

  // Row/column operator.
  template <typename __T = type, typename = std::enable_if_t<__T::is2D()>>
  auto row(int __i);
  template <typename __T = type, typename = std::enable_if_t<__T::is2D()>>
  auto column(int __i);

  // Unary operators.
  type &operator++();
  type operator++(int);
  type &operator--();
  type operator--(int);

  // Binary operators.
  auto operator +(const type &__RHS) const;
  auto operator -(const type &__RHS) const;
  auto operator *(const type &__RHS) const;
  auto operator /(const type &__RHS) const;
  type operator &(const type &__RHS) const;
  type operator |(const type &__RHS) const;
  type operator ^(const type &__RHS) const;

  // Compound assignment.
  type &operator +=(const type &__RHS);
  type &operator -=(const type &__RHS);
  type &operator *=(const type &__RHS);
  type &operator /=(const type &__RHS);
  type &operator &=(const type &__RHS);
  type &operator |=(const type &__RHS);
  type &operator ^=(const type &__RHS);

  // Compare operators.
  simd<uint16_t, __N> operator >(const type &__RHS) const;
  simd<uint16_t, __N> operator >=(const type &__RHS) const;
  simd<uint16_t, __N> operator <(const type &__RHS) const;
  simd<uint16_t, __N> operator <=(const type &__RHS) const;
  simd<uint16_t, __N> operator ==(const type &__RHS) const;
  simd<uint16_t, __N> operator !=(const type &__RHS) const;

  // Select operations.
  template <int __Size, int __Stride, typename __T = type,
            typename = std::enable_if_t<__T::is1D()>>
  auto select(uint16_t __Offset = 0) &;
  template <int __Size, int __Stride, typename __T = type,
            typename = std::enable_if_t<__T::is1D()>>
  auto select(uint16_t __Offset = 0) &&;
  template <int __SizeY, int __StrideY, int __SizeX, int __StrideX,
            typename __T = type, typename = std::enable_if_t<__T::is2D()>>
  auto select(uint16_t __OffsetY = 0, uint16_t __OffsetX = 0) &;
  template <int __SizeY, int __StrideY, int __SizeX, int __StrideX,
            typename __T = type, typename = std::enable_if_t<__T::is2D()>>
  auto select(uint16_t __OffsetY = 0, uint16_t __OffsetX = 0) &&;

  // Replicate operations.
  template <int Rep> simd<element_type, Rep> replicate();
  template <int Rep, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX);
  template <int Rep, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY,
                                        uint16_t OffsetX);
  template <int Rep, int VS, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX);
  template <int Rep, int VS, int W>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY,
                                        uint16_t OffsetX);
  template <int Rep, int VS, int W, int HS>
  simd<element_type, Rep * W> replicate(uint16_t OffsetX);
  template <int Rep, int VS, int W, int HS>
  simd<element_type, Rep * W> replicate(uint16_t OffsetY,
                                        uint16_t OffsetX);

  // Format operations.
  template <typename __EltTy> auto format();
  template <typename __EltTy> auto format() &&;
  template <typename __EltTy, int __Height, int __Width> auto format() &;
  template <typename __EltTy, int __Height, int __Width> auto format() &&;

  // Merge operations.
  void merge(const value_type &__Val, const __mask_type_t<length> &__Mask);
  void merge(const value_type &__Val1, value_type __Val2,
             const __mask_type_t<length> &__Mask);
} // namespace gpu
} // namespace intel
} // namespace sycl
```

```sycl::intel::gpu::simd_view``` class supports all the element-wise operations and
other utility functions defined for ```sycl::intel::gpu::simd``` class. It also
provides region accessors and more generic operations tailored for 2D regions,
such as row/column operators and 2D select/replicate/format/merge operations.

```cpp
  simd<float, 32> v1;
  auto m1 = v1.format<float, 4, 8>();
  simd<float, 4> v2;
  auto m2 = v2.format<float, 2, 2>();

  // ...
  m2 = m1.select<2, 2, 2, 4>(1, 2);  // v_size = 2, v_stride = 2,
                                     // h_size = 2, h_stride = 4,
                                     // v_offset = 1, h_offset = 2.
```
<p align="center">
<img src="images/Matrix_2_2_2_4__1_2.svg" title="2D select example" width="700" height="200"/>
</p>

```cpp
  m1.select<4, 1, 4, 2>(0, 0) = 0.0f; // selected elements of m1
                                      // are replaced with 0.0f
```

<p align="center">
<img src="images/Matrix_4_1_4_2__0_0.svg" title="2D select example" width="700" height="200"/>
</p>

### Reduction functions

Explicit SIMD provides the following reduction functions for simd objects.
Compiler will produce optimal code sequence on the target device to apply the
specified operation to all scalar elements in the input simd vector. Note that
the order of element-wise operations is not guaranteed and the correctness of
result should not depend on a particular computation order.
- ```cm_sum(simd<T1, SZ> v)```: returns a scalar value of type **T0** that's
equal to the sum of all data elements of type **T1** in simd vector **v** of
length **SZ**.
- ```cm_prod(simd<T1, SZ> v)```: returns a scalar value of type **T0** that's
equal to the product of all data elements of type **T1** in simd vector **v** of
length **SZ**.
- ```cm_reduced_max(simd<T1, SZ> v)```: returns a scalar value of type **T0**
that's equal to the maximum value of all data elements of type **T1** in simd
vector **v** of length **SZ**.
- ```cm_reduced_min(simd<T1, SZ> v)```: returns a scalar value of type **T0**
that's equal to the minimum value of all data elements of type **T1** in simd
vector **v** of length **SZ**.

```cpp
template <typename T0, typename T1, int SZ> T0 cm_sum(simd<T1, SZ> v);
template <typename T0, typename T1, int SZ> T0 cm_prod(simd<T1, SZ> v);
template <typename T0, typename T1, int SZ> T0 cm_reduced_max(simd<T1, SZ> v);
template <typename T0, typename T1, int SZ> T0 cm_reduced_min(simd<T1, SZ> v);
```

### Memory access APIs

Currently the variety of memory objects supported by the Explicit SIMD extension
implementation is limited to the following:
- USM pointers (aka 'flat address')
- 1D global accessors
- 2D image accessors

The implementation further limits the set of memory access operations
which a kernel can perform through those memory objects. Basically, all of them
are special intrinsic APIs described below rather than standard SYCL accessor or
pointer operations. Examples of unsupported features include
`accessor::get_pointer()`, accessor's `operator []`, C/C++ dereference of an
USM pointer, local accessors, 2D and 3D accessors.
Those are temporary restrictions, to be dropped in future.

See auto-generated documentation for the complete list of APIs here. (TBD)

#### USM pointer-based memory access
##### Flat-address gather/scatter
perform scattered read/write for the given starting pointer **p** and
**offsets**.

```cpp
template <typename T, int n, int ElemsPerAddr = 1,
	  CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
typename std::enable_if<((n == 16 || n == 32) &&
    (ElemsPerAddr == 1 || ElemsPerAddr == 2 || ElemsPerAddr == 4)),
    simd<T, n*ElemsPerAddr>>::type
flat_load(T *p, simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1);

template <typename T, int n, int ElemsPerAddr = 1,
	  CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
typename std::enable_if<((n == 16 || n == 32) &&
    (ElemsPerAddr == 1 || ElemsPerAddr == 2 || ElemsPerAddr == 4)),
    void>::type
flat_store(T *p, simd<T, n*ElemsPerAddr> vals, simd<uint32_t, n> offsets,
           simd<uint16_t, n> pred = 1);
```
##### Flat-address block load/store
read or write a consecutive block of data for the memory location specified by
**addr**.

```cpp
template <typename T, int n, CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
simd<T, n> flat_block_load(const T *const addr);

template <typename T, int n, CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
void flat_block_store(T *p, simd<T, n> vals);
```

##### Flat-address load4/store4
perform scattered read/write for the given starting pointer **p** and
**offsets**. Up to 4 data elements may be accessed at each address depending on
the enabled channel **Mask**.

```cpp
template <typename T, int n, ChannelMaskType Mask,
          CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
    typename std::enable_if<(n == 16 || n == 32),
                            simd<T, n * __NumChannels(Mask)>>::type
    flat_load4(T *p, simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1);

template <typename T, int n, ChannelMaskType Mask,
          CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
typename std::enable_if<(n == 16 || n == 32), void>::type
    flat_store4(T *p, simd<T, n * __NumChannels(Mask)> vals,
            simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1);
```

##### Flat-address atomic inc/dec
perform atomic memory access operation with zero source operand.

```cpp
template <CmAtomicOpType Op, typename T, int n,
          CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
    typename std::enable_if<__check_atomic<Op, T, n, 0>(), simd<T, n>>::type
    flat_atomic(T *p, simd<unsigned, n> offset, simd<ushort, n> pred);
```

- ```Flat-address atomic add/sub/min/max/etc.```: perform atomic memory
access operation with one source operand.

```cpp
template <CmAtomicOpType Op, typename T, int n,
          CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
    typename std::enable_if<__check_atomic<Op, T, n, 1>(), simd<T, n>>::type
    flat_atomic(T *p, simd<unsigned, n> offset, simd<T, n> src0,
                simd<ushort, n> pred);
```

- ```Flat-address atomic CMPXCHG```: perform atomic memory access operation
with two source operands.

```cpp

template <CmAtomicOpType Op, typename T, int n,
          CacheHint L1H = CacheHint::Default,
          CacheHint L3H = CacheHint::Default>
    typename std::enable_if<__check_atomic<Op, T, n, 2>(), simd<T, n>>::type
    flat_atomic(T *p, simd<unsigned, n> offset, simd<T, n> src0,
                simd<T, n> src1, simd<ushort, n> pred);
```

#### Accessor-based memory access

Examples:
##### 1D surface block store.
T - element type, n - vector length, acc - SYCL global buffer accessor,
offset - byte offset in the buffer, vals - vector value to store
```cpp
template <typename T, int n, typename AccessorTy>
void block_store(AccessorTy acc, uint32_t offset, simd<T, n> vals);
```
##### 2D media block load.
T - element type, m and n - block dimensions, acc - SYCL image2D accessor,
x and y - image coordinates
```cpp
template <typename T, int m, int n, typename AccessorTy, unsigned plane = 0>
simd<T, m * n> media_block_load(AccessorTy acc, unsigned x, unsigned y);
```

#### Local address space allocation and access
Examples:

##### SLM scatter.
T - element type, n - vector length (16 or 32), vals - vector value to store,
offsets - byte offsets in the local memory, pred - store mask
```cpp
template <typename T, int n>
typename std::enable_if<(n == 16 || n == 32), void>::type
slm_store(simd<T, n> vals, simd<uint32_t, n> offsets, simd<uint16_t, n> pred = 1);
```
<br>

### Private Global Variables.

Explicit SIMD extension supports "private global" variables - file scope
variables in private address space (similar to thread-local variables on host).
These variables have 1 copy per work-item (which maps to a single SIMD thread in
ESP) and are visible to all functions in the translation unit. Conceptually they
map to SPIRV variable with private storage class. Private globals can be bound
to a specific byte offset within the GRF. To mark a file scope variable as
private global, the `INTEL_GPU_PRIVATE` attribute is used,
`INTEL_GPU_REGISTER` is used to bind it the register file:

```cpp
INTEL_GPU_PRIVATE INTEL_GPU_REGISTER(32) simd<int, 16> vc;
```
<br>

## Examples
### Vector addition (USM)
```cpp
#include <iostream>
#include <CL/sycl.hpp>
#include <sycl_[[sycl_explicit_simd]].hpp>

using namespace cl::sycl;

int main(void) {
  constexpr unsigned Size = 128;
  constexpr unsigned VL = 32;
  constexpr unsigned GroupSize = 2;

  queue q;
  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();
  float* A = static_cast<float*>(malloc_shared(Size*sizeof(float), dev, ctxt));
  float* B = static_cast<float*>(malloc_shared(Size*sizeof(float), dev, ctxt));
  float* C = static_cast<float*>(malloc_shared(Size*sizeof(float), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
  }

  // iteration space
  nd_range<1> Range(range<1>(Size/VL), range<1>(GroupSize));

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(
      Range, [=](nd_item<1> i) [[sycl_explicit_simd]] {

      auto offset = i.get_global_id(0) * VL;
      sycl::intel::gpu<float, VL> va = sycl::intel::gpu::flat_block_load<float, VL>(A + offset);
      sycl::intel::gpu<float, VL> vb = sycl::intel::gpu::flat_block_load<float, VL>(B + offset);
      sycl::intel::gpu<float, VL> vc = va + vb;
      sycl::intel::gpu::flat_block_store<float, VL>(C + offset, vc);
    });
  });
  e.wait();
  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
          << " + " << B[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: " << ((float)(Size - err_cnt) / (float)Size)*100.0f <<
      "% (" << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
```

### Open questions
- Vectorization controls (e.g. enforcing vector length generated by the
  compiler) for vector/matrix operations.
- Enabling loop vectorizer on inner loops in a Explicit SIMD kernel or function.

### TODOs

- Design interoperability with SPMD context
- Generate sycl::intel::gpu API documentation from sources
- Section covering 2D use cases
- A bridge from `std::simd` to `sycl::intel::gpu::simd`
- Describe `simd_view` class restrictions
- Consider auto-inclusion of sycl_[[sycl_explicit_simd]].hpp under -fsycl-esimd option
