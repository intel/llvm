#ifndef SYCL_HPP
#define SYCL_HPP

#define __SYCL_TYPE(x) [[__sycl_detail__::sycl_type(x)]]
#define __SYCL_BUILTIN_ALIAS(X) [[clang::builtin_alias(X)]]

// Shared code for SYCL tests

namespace sycl {
inline namespace _V1 {
namespace access {

enum class target {
  global_buffer = 2014,
  constant_buffer,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class mode {
  read = 1024,
  write,
  read_write,
  discard_write,
  discard_read_write,
  atomic
};

enum class placeholder { false_t,
                         true_t };

enum class address_space : int {
  private_space = 0,
  global_space,
  constant_space,
  local_space,
  generic_space
};

enum class decorated : int {
  no = 0,
  yes,
  legacy,
};
} // namespace access

// Dummy aspect enum with limited enumerators
enum class __SYCL_TYPE(aspect) aspect {
  host = 0,
  cpu = 1,
  gpu = 2,
  accelerator = 3,
  custom = 4,
  fp16 = 5,
  fp64 = 6,
};

class property_list {};

namespace ext {
namespace intel {
namespace property {
struct __SYCL_TYPE(buffer_location) buffer_location {
  template <int> class instance {};
};
} // namespace property
} // namespace intel
} // namespace ext

namespace ext {
namespace oneapi {
template <typename... properties>
class __SYCL_TYPE(accessor_property_list) accessor_property_list {};

// device_global type decorated with attributes
template <typename T>
struct [[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] device_global {
public:
  const T &get() const noexcept { return *Data; }
  device_global() {}
  operator T &() noexcept { return *Data; }

private:
  T *Data;
};
} // namespace oneapi
} // namespace ext

namespace detail {
namespace half_impl {
struct half {
#ifdef __SYCL_DEVICE_ONLY
  _Float16 data;
#else
  char data[2];
#endif
};
} // namespace half_impl
} // namespace detail

using half = detail::half_impl::half;

template <int dim>
struct range {
};

template <int dim>
struct id {
};

template <int dim>
struct _ImplT {
  range<dim> AccessRange;
  range<dim> MemRange;
  id<dim> Offset;
};

template <typename dataT, access::target accessTarget>
struct DeviceValueType;

template <typename dataT>
struct DeviceValueType<dataT, access::target::global_buffer> {
  using type = __attribute__((opencl_global)) dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::constant_buffer> {
  using type = __attribute__((opencl_constant)) dataT;
};

template <typename dataT>
struct DeviceValueType<dataT, access::target::local> {
  using type = __attribute__((opencl_local)) dataT;
};

template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t,
          typename propertyListT = ext::oneapi::accessor_property_list<>>
class __attribute__((sycl_special_class)) __SYCL_TYPE(accessor) accessor {
public:
  void use(void) const {}
  void use(void *) const {}
  _ImplT<dimensions> impl;

private:
  using PtrType = typename DeviceValueType<dataT, accessTarget>::type *;
  void __init(PtrType Ptr, range<dimensions> AccessRange,
              range<dimensions> MemRange, id<dimensions> Offset) {}
  void __init_esimd(PtrType Ptr) {}
  friend class stream;
};

template <int dimensions, access::mode accessmode, access::target accesstarget>
struct opencl_image_type;

#define IMAGETY_DEFINE(dim, accessmode, amsuffix, Target, ifarray_) \
  template <>                                                       \
  struct opencl_image_type<dim, access::mode::accessmode,           \
                           access::target::Target> {                \
    using type = __ocl_image##dim##d_##ifarray_##amsuffix##_t;      \
  };

#ifdef __SYCL_DEVICE_ONLY__

#define IMAGETY_READ_3_DIM_IMAGE       \
  IMAGETY_DEFINE(1, read, ro, image, ) \
  IMAGETY_DEFINE(2, read, ro, image, ) \
  IMAGETY_DEFINE(3, read, ro, image, )

#define IMAGETY_WRITE_3_DIM_IMAGE       \
  IMAGETY_DEFINE(1, write, wo, image, ) \
  IMAGETY_DEFINE(2, write, wo, image, ) \
  IMAGETY_DEFINE(3, write, wo, image, )

#define IMAGETY_READ_2_DIM_IARRAY                  \
  IMAGETY_DEFINE(1, read, ro, image_array, array_) \
  IMAGETY_DEFINE(2, read, ro, image_array, array_)

#define IMAGETY_WRITE_2_DIM_IARRAY                  \
  IMAGETY_DEFINE(1, write, wo, image_array, array_) \
  IMAGETY_DEFINE(2, write, wo, image_array, array_)

IMAGETY_READ_3_DIM_IMAGE
IMAGETY_WRITE_3_DIM_IMAGE

IMAGETY_READ_2_DIM_IARRAY
IMAGETY_WRITE_2_DIM_IARRAY

#endif // __SYCL_DEVICE_ONLY__

template <int dim, access::mode accessmode, access::target accesstarget>
struct _ImageImplT {
#ifdef __SYCL_DEVICE_ONLY__
  typename opencl_image_type<dim, accessmode, accesstarget>::type MImageObj;
#else
  range<dim> AccessRange;
  range<dim> MemRange;
  id<dim> Offset;
#endif
};

template <typename dataT, int dimensions, access::mode accessmode>
class __attribute__((sycl_special_class))  __SYCL_TYPE(accessor) accessor<dataT, dimensions, accessmode, access::target::image, access::placeholder::false_t> {
public:
  void use(void) const {}
  template <typename... T>
  void use(T... args) {}
  template <typename... T>
  void use(T... args) const {}
  _ImageImplT<dimensions, accessmode, access::target::image> impl;
#ifdef __SYCL_DEVICE_ONLY__
  void __init(typename opencl_image_type<dimensions, accessmode, access::target::image>::type ImageObj) { impl.MImageObj = ImageObj; }
#endif
};

template <typename dataT, int dimensions>
class __attribute__((sycl_special_class)) __SYCL_TYPE(local_accessor)
local_accessor: public accessor<dataT,
        dimensions, access::mode::read_write,
        access::target::local> {
public:
  void use(void) const {}
  template <typename... T>
  void use(T... args) {}
  template <typename... T>
  void use(T... args) const {}
  _ImplT<dimensions> impl;

private:
#ifdef __SYCL_DEVICE_ONLY__
  void __init(__attribute__((opencl_local)) dataT *Ptr, range<dimensions> AccessRange,
              range<dimensions> MemRange, id<dimensions> Offset) {}
#endif
};

struct sampler_impl {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_sampler_t m_Sampler;
#endif
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(sampler) sampler {
  struct sampler_impl impl;
#ifdef __SYCL_DEVICE_ONLY__
  void __init(__ocl_sampler_t Sampler) { impl.m_Sampler = Sampler; }
#endif

public:
  void use(void) const {}
};

class event {};
class queue {
public:
  template <typename T>
  event submit(T cgf) { return event{}; }
};
class auto_name {};
template <typename Name, typename Type>
struct get_kernel_name_t {
  using name = Name;
};
template <typename Type>
struct get_kernel_name_t<auto_name, Type> {
  using name = Type;
};

template <int dimensions = 1>
class __SYCL_TYPE(group) group {
public:
  group() = default; // fake constructor
};

class __SYCL_TYPE(kernel_handler) kernel_handler {
  void __init_specialization_constants_buffer(char *specialization_constants_buffer) {}
};

template <typename T> class __SYCL_TYPE(specialization_id) specialization_id {
public:
  using value_type = T;

  template <class... Args>
  explicit constexpr specialization_id(Args &&...args)
      : MDefaultValue(args...) {}

  specialization_id(const specialization_id &rhs) = delete;
  specialization_id(specialization_id &&rhs) = delete;
  specialization_id &operator=(const specialization_id &rhs) = delete;
  specialization_id &operator=(specialization_id &&rhs) = delete;

  T getDefaultValue() const { return MDefaultValue; }

private:
  T MDefaultValue;
};

// Used when parallel_for range is rounded-up.
template <typename Type> class __pf_kernel_wrapper;

template <typename Type> struct get_kernel_wrapper_name_t {
  using name =
      __pf_kernel_wrapper<typename get_kernel_name_t<auto_name, Type>::name>;
};

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))
template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_single_task(const KernelType &kernelFunc) { // #KernelSingleTaskFunc
#ifdef __SYCL_DEVICE_ONLY__
  kernelFunc(); // #KernelSingleTaskKernelFuncCall
#else
  (void)kernelFunc;
#endif
}
template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_single_task(const KernelType &kernelFunc, kernel_handler kh) {
#ifdef __SYCL_DEVICE_ONLY__
  kernelFunc(kh);
#else
  (void)kernelFunc;
#endif
}
template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_parallel_for(const KernelType &kernelFunc) {
#ifdef __SYCL_DEVICE_ONLY__
  kernelFunc();
#else
  (void)kernelFunc;
#endif
}
template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_parallel_for_work_group(const KernelType &KernelFunc, kernel_handler kh) {
#ifdef __SYCL_DEVICE_ONLY__
  KernelFunc(group<1>(), kh);
#else
  (void)KernelFunc;
#endif
}

class handler {
public:
  template <typename KernelName = auto_name, typename KernelType>
  void single_task(const KernelType &kernelFunc) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
    kernel_single_task<NameT>(kernelFunc); // #KernelSingleTask
  }
  template <typename KernelName = auto_name, typename KernelType>
  void single_task(const KernelType &kernelFunc, kernel_handler kh) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
    kernel_single_task<NameT>(kernelFunc, kh);
  }
  template <typename KernelName = auto_name, typename KernelType>
  void parallel_for(const KernelType &kernelObj) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
    using NameWT = typename get_kernel_wrapper_name_t<NameT>::name;
    kernel_parallel_for<NameT>(kernelObj);
  }
  template <typename KernelName = auto_name, typename KernelType>
  void parallel_for_work_group(const KernelType &kernelFunc, kernel_handler kh) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
    kernel_parallel_for_work_group<NameT>(kernelFunc, kh);
  }
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(stream) stream {
  accessor<int, 1, access::mode::read> acc;

public:
  stream(unsigned long BufferSize, unsigned long MaxStatementSize,
         handler &CGH) {}
#ifdef __SYCL_DEVICE_ONLY__
  // Default constructor for objects later initialized with __init member.
  stream() = default;
#endif

  void __init(__attribute((opencl_global)) char *Ptr, range<1> AccessRange,
              range<1> MemRange, id<1> Offset, int _FlushBufferSize) {
    Acc.__init(Ptr, AccessRange, MemRange, Offset);
    FlushBufferSize = _FlushBufferSize;
  }

  void use() const {}

  void __finalize() {}

private:
  sycl::accessor<char, 1, sycl::access::mode::read_write> Acc;
  int FlushBufferSize;
};

template <typename ElementType, access::address_space addressSpace>
struct DecoratedType;

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::private_space> {
  using type = __attribute__((opencl_private)) ElementType;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::generic_space> {
  using type = ElementType;
};

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::global_space> {
  using type = __attribute__((opencl_global)) ElementType;
};

template <typename T, access::address_space AS,
          access::decorated DecorateAddress = access::decorated::legacy>
class __SYCL_TYPE(multi_ptr) multi_ptr {
  using decorated_type = typename DecoratedType<T, AS>::type;

  static_assert(DecorateAddress != access::decorated::legacy);
  static_assert(AS != access::address_space::constant_space);

public:
  using pointer = decorated_type *;

  multi_ptr(typename multi_ptr<T, AS, access::decorated::yes>::pointer Ptr)
    : m_Pointer((pointer)(Ptr)) {}
  pointer get() { return m_Pointer; }

 private:
  pointer m_Pointer;
};

template <typename ElementType, access::address_space Space>
struct LegacyPointerType {
  using pointer_t = typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
};

// Legacy specialization
template <typename T, access::address_space AS>
class __SYCL_TYPE(multi_ptr) multi_ptr<T, AS, access::decorated::legacy> {
public:
  using pointer_t = typename LegacyPointerType<T, AS>::pointer_t;

  multi_ptr(typename multi_ptr<T, AS, access::decorated::yes>::pointer Ptr)
    : m_Pointer((pointer_t)(Ptr)) {}
  multi_ptr(T *Ptr) : m_Pointer((pointer_t)(Ptr)) {} // #MultiPtrConstructor
  pointer_t get() { return m_Pointer; }

private:
  pointer_t m_Pointer;
};

namespace ext {
namespace oneapi {
namespace experimental {
template <typename T, typename... Props>
class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_arg) annotated_arg {
  T obj;
  #ifdef __SYCL_DEVICE_ONLY__
    void __init(T _obj) {}
  #endif
};

template <typename T, typename... Props>
class __attribute__((sycl_special_class)) __SYCL_TYPE(annotated_ptr) annotated_ptr {
  T* obj;
  #ifdef __SYCL_DEVICE_ONLY__
    void __init(T* _obj) {}
  #endif
};

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // namespace _V1
} // namespace sycl

#endif
