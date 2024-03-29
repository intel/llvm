#pragma once

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))
#define __SYCL_TYPE(x) [[__sycl_detail__::sycl_type(x)]]
#define __SYCL_BUILTIN_ALIAS(X) [[clang::builtin_alias(X)]]

#ifdef SYCL_EXTERNAL
#define __DPCPP_SYCL_EXTERNAL SYCL_EXTERNAL
#else
#ifdef __SYCL_DEVICE_ONLY__
#define __DPCPP_SYCL_EXTERNAL __attribute__((sycl_device))
#else
#define __DPCPP_SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif
#endif

extern "C" int printf(const char* fmt, ...);

#ifdef __SYCL_DEVICE_ONLY__
__attribute__((convergent)) extern __attribute__((sycl_device)) void
__spirv_ControlBarrier(int, int, int) noexcept;
#endif

// Dummy runtime classes to model SYCL API.
namespace sycl {
inline namespace _V1 {
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

template <int dimensions = 1>
class __SYCL_TYPE(group) group {
public:
  group() = default; // fake constructor
  // Dummy parallel_for_work_item function to mimic calls from
  // parallel_for_work_group.
  void parallel_for_work_item() {
  }
};

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

enum class placeholder {
  false_t,
  true_t
};

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
  legacy
};
} // namespace access

// Dummy aspect enum with limited enumerators
enum class __SYCL_TYPE(aspect) aspect { // #AspectEnum
  host = 0,
  cpu = 1,
  gpu = 2,
  accelerator = 3,
  custom = 4,
  fp16 = 5,
  fp64 = 6,
};

using access::target;
using access_mode = access::mode;

namespace property {

enum prop_type {
  use_host_ptr = 0,
  use_mutex,
  context_bound,
  enable_profiling,
  base_prop
};

struct property_base {
  virtual prop_type type() const = 0;
};
} // namespace property

class property_list {
public:
  template <typename... propertiesTN>
  property_list(propertiesTN... props){};

  template <typename propertyT>
  bool has_property() const { return true; }

  template <typename propertyT>
  propertyT get_property() const {
    return propertyT{};
  }

  bool operator==(const property_list &rhs) const { return false; }

  bool operator!=(const property_list &rhs) const { return false; }
};

namespace ext {
namespace intel {
namespace property {
// Compile time known accessor property
struct __SYCL_TYPE(buffer_location) buffer_location {
  template <int> class instance {};
};
} // namespace property
} // namespace intel
} // namespace ext

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

template <typename ElementType>
struct DecoratedType<ElementType, access::address_space::constant_space> {
#if defined(RESTRICT_WRITE_ACCESS_TO_CONSTANT_PTR)
  using type = const __attribute__((opencl_global)) ElementType;
#else
  using type = __attribute__((opencl_global)) ElementType;
#endif
};

// Equivalent to std::conditional
template <bool B, class T, class F>
struct conditional { using type = T; };

template <class T, class F>
struct conditional<false, T, F> { using type = F; };

template <bool B, class T, class F>
using conditional_t = typename conditional<B, T, F>::type;

template <typename T, access::address_space AS,
          access::decorated DecorateAddress = access::decorated::legacy>
class __SYCL_TYPE(multi_ptr) multi_ptr {
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;

  using decorated_type = typename DecoratedType<T, AS>::type;

  static_assert(DecorateAddress != access::decorated::legacy);
  static_assert(AS != access::address_space::constant_space);

public:
  using pointer = conditional_t<is_decorated, decorated_type *, T *>;

  multi_ptr(typename multi_ptr<T, AS, access::decorated::yes>::pointer Ptr)
    : m_Pointer((pointer)(Ptr)) {} // #MultiPtrConstructor
  pointer get() { return m_Pointer; }

 private:
  pointer m_Pointer;
};

template <typename ElementType, access::address_space Space>
struct LegacyPointerType {
  using pointer_t = typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
};

template <typename ElementType>
struct LegacyPointerType<ElementType, access::address_space::constant_space> {
  using decorated_type = typename DecoratedType<ElementType, access::address_space::constant_space>::type;
  using pointer_t = decorated_type *;
};

// Legacy specialization
template <typename T, access::address_space AS>
class __SYCL_TYPE(multi_ptr) multi_ptr<T, AS, access::decorated::legacy> {
public:
  using pointer_t = typename LegacyPointerType<T, AS>::pointer_t;

  multi_ptr(typename multi_ptr<T, AS, access::decorated::yes>::pointer Ptr)
    : m_Pointer((pointer_t)(Ptr)) {}
  multi_ptr(T *Ptr) : m_Pointer((pointer_t)(Ptr)) {} // #LegacyMultiPtrConstructor
  pointer_t get() { return m_Pointer; }

 private:
  pointer_t m_Pointer;
};
         
namespace ext {
namespace oneapi {
namespace property {
// Compile time known accessor property
struct __SYCL_TYPE(no_alias) no_alias {
  template <bool> class instance {};
};
} // namespace property

// device_global type decorated with attributes
template <typename T>
class [[__sycl_detail__::device_global]] [[__sycl_detail__::global_variable_allowed]] device_global {
public:
  const T &get() const noexcept { return *Data; }
  device_global() {}
  operator T &() noexcept { return *Data; }

private:
  T *Data;
};

} // namespace oneapi
} // namespace ext

namespace ext {
namespace intel {
namespace experimental {

// host_pipe class decorated with attribute
template <class _name, class _dataT>
class
host_pipe {

public:
  struct
#ifdef __SYCL_DEVICE_ONLY__
  [[__sycl_detail__::sycl_type(host_pipe)]]
#endif
  __pipeType { const char __p; };

  static constexpr __pipeType __pipe = {0};
  static _dataT read() {
    (void)__pipe;
  }
};

} // namespace experimental
} // namespace intel
} // namespace ext

template <int dim>
struct id {
  template <typename... T>
  id(T... args) {} // fake constructor
private:
  // Some fake field added to see using of id arguments in the
  // kernel wrapper
  int Data;
};

template <int dim> struct item {
  template <typename... T>
  item(T... args) {} // fake constructor
private:
  // Some fake field added to see using of item arguments in the
  // kernel wrapper
  int Data;
};

namespace ext {
namespace oneapi {
template <typename... properties>
class __SYCL_TYPE(accessor_property_list) accessor_property_list {};
} // namespace oneapi
} // namespace ext

template <int dim>
struct range {
  template <typename... T>
  range(T... args) {} // fake constructor
private:
  // Some fake field added to see using of range arguments in the
  // kernel wrapper
  int Data;
};

template <int dim>
struct nd_range {
};

template <int dim>
struct _ImplT {
  range<dim> AccessRange;
  range<dim> MemRange;
  id<dim> Offset;
};

template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget = access::target::global_buffer,
          access::placeholder isPlaceholder = access::placeholder::false_t,
          typename propertyListT = ext::oneapi::accessor_property_list<>>
class __attribute__((sycl_special_class)) __SYCL_TYPE(accessor) accessor {

public:
  void use(void) const {}
  template <typename... T>
  void use(T... args) {}
  template <typename... T>
  void use(T... args) const {}
  _ImplT<dimensions> impl;

private:
  void __init(__attribute__((opencl_global)) dataT *Ptr, range<dimensions> AccessRange,
              range<dimensions> MemRange, id<dimensions> Offset) {}
  void __init_esimd(__attribute__((opencl_global)) dataT *Ptr) {}
  friend class stream;
};

template <int dimensions, access::mode accessmode, access::target accesstarget>
struct opencl_image_type;

#ifdef __SYCL_DEVICE_ONLY__
#define IMAGETY_DEFINE(dim, accessmode, amsuffix, Target, ifarray_) \
  template <>                                                       \
  struct opencl_image_type<dim, access::mode::accessmode,           \
                           access::target::Target> {                \
    using type = __ocl_image##dim##d_##ifarray_##amsuffix##_t;      \
  };

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

#endif

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
class __attribute__((sycl_special_class)) __SYCL_TYPE(accessor) accessor<dataT, dimensions, accessmode, access::target::image, access::placeholder::false_t> {
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

template <typename dataT, int dimensions, access::mode accessmode>
class accessor<dataT, dimensions, accessmode, access::target::host_image, access::placeholder::false_t> {
public:
  void use(void) const {}
  template <typename... T>
  void use(T... args) {}
  template <typename... T>
  void use(T... args) const {}
  _ImageImplT<dimensions, accessmode, access::target::host_image> impl;
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

// TODO: Add support for image_array accessor.
// template <typename dataT, int dimensions, access::mode accessmode>
//class accessor<dataT, dimensions, accessmode, access::target::image_array, access::placeholder::false_t>

class kernel {};
class context {};
class device {};
class event {};

class queue {
public:
  template <typename T>
  event submit(T cgf) { return event{}; }

  void wait() {}
  void wait_and_throw() {}
  void throw_asynchronous() {}
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

namespace ext {
namespace oneapi {
namespace experimental {
#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
#define __SYCL_CONSTANT_AS
#endif
template <typename... Args>
int printf(const __SYCL_CONSTANT_AS char *__format, Args... args) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  return __spirv_ocl_printf(__format, args...);
#else
  return ::printf(__format, args...);
#endif // defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
}

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

#if __cplusplus >= 201703L
template<typename T> specialization_id(T) -> specialization_id<T>;
#endif // C++17.

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))
template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_single_task(const KernelType &kernelFunc) { // #KernelSingleTask
  kernelFunc();
}

template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_single_task(const KernelType &kernelFunc, kernel_handler kh) {
  kernelFunc(kh);
}

template <typename KernelName, typename KernelType>
ATTR_SYCL_KERNEL void kernel_single_task_2017(KernelType kernelFunc) { // #KernelSingleTask2017
  kernelFunc();
}

template <typename KernelName, typename KernelType, int Dims>
ATTR_SYCL_KERNEL void
kernel_parallel_for(const KernelType &KernelFunc) {
  KernelFunc(id<Dims>());
}

// Dummy parallel_for_work_item function to mimic calls from
// parallel_for_work_group.
void parallel_for_work_item() {
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrier(0, 0, 0);
#endif
}

template <typename KernelName, typename KernelType, int Dims>
ATTR_SYCL_KERNEL void
kernel_parallel_for_work_group(const KernelType &KernelFunc) {
  KernelFunc(group<Dims>());
  parallel_for_work_item();
}

class handler {
public:
  template <typename KernelName = auto_name, typename KernelType, int Dims>
  void parallel_for(range<Dims> numWorkItems, const KernelType &kernelFunc) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for<NameT, KernelType, Dims>(kernelFunc);
#else
    kernelFunc();
#endif
  }

  template <typename KernelName = auto_name, typename KernelType, int Dims>
  void parallel_for_work_group(range<Dims> numWorkGroups, range<Dims> WorkGroupSize, const KernelType &kernelFunc) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_parallel_for_work_group<NameT, KernelType, Dims>(kernelFunc);
#else
    group<Dims> G;
    kernelFunc(G);
#endif
  }

  template <typename KernelName = auto_name, typename KernelType>
  void single_task(const KernelType &kernelFunc) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<NameT>(kernelFunc);
#else
    kernelFunc();
#endif
  }

  template <typename KernelName = auto_name, typename KernelType>
  void single_task(const KernelType &kernelFunc, kernel_handler kh) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task<NameT>(kernelFunc, kh);
#else
    kernelFunc(kh);
#endif
  }

  template <typename KernelName = auto_name, typename KernelType>
  void single_task_2017(KernelType kernelFunc) {
    using NameT = typename get_kernel_name_t<KernelName, KernelType>::name;
#ifdef __SYCL_DEVICE_ONLY__
    kernel_single_task_2017<NameT>(kernelFunc);
#else
    kernelFunc();
#endif
  }
};

class __attribute__((sycl_special_class)) __SYCL_TYPE(stream) stream {
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

  void __finalize() {}

private:
  sycl::accessor<char, 1, sycl::access::mode::read_write> Acc;
  int FlushBufferSize;
};

template <typename T>
const stream& operator<<(const stream &S, T&&) {
  return S;
}

template <typename T, int dimensions = 1,
          typename AllocatorT = int /*fake type as AllocatorT is not used*/>
class buffer {
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  template <typename... ParamTypes>
  buffer(ParamTypes... args) {} // fake constructor

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {}) {}

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {}) {}

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {}) {}

  buffer(const buffer &rhs) = default;

  buffer(buffer &&rhs) = default;

  buffer &operator=(const buffer &rhs) = default;

  buffer &operator=(buffer &&rhs) = default;

  ~buffer() = default;

  range<dimensions> get_range() const { return range<dimensions>{}; }

  template <access::mode mode,
            access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target, access::placeholder::false_t>
  get_access(handler &commandGroupHandler) {
    return accessor<T, dimensions, mode, target, access::placeholder::false_t>{};
  }

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer,
           access::placeholder::false_t>
  get_access() {
    return accessor<T, dimensions, mode, access::target::host_buffer,
                    access::placeholder::false_t>{};
  }

  template <typename Destination>
  void set_final_data(Destination finalData = nullptr) {}
};

enum class image_channel_order : unsigned int {
  a,
  r,
  rx,
  rg,
  rgx,
  ra,
  rgb,
  rgbx,
  rgba,
  argb,
  bgra,
  intensity,
  luminance,
  abgr
};

enum class image_channel_type : unsigned int {
  snorm_int8,
  snorm_int16,
  unorm_int8,
  unorm_int16,
  unorm_short_565,
  unorm_short_555,
  unorm_int_101010,
  signed_int8,
  signed_int16,
  signed_int32,
  unsigned_int8,
  unsigned_int16,
  unsigned_int32,
  fp16,
  fp32
};

template <int dimensions = 1, typename AllocatorT = int>
class image {
public:
  image(image_channel_order Order, image_channel_type Type,
        const range<dimensions> &Range,
        const property_list &PropList = {}) {}

  /* -- common interface members -- */

  image(const image &rhs) = default;

  image(image &&rhs) = default;

  image &operator=(const image &rhs) = default;

  image &operator=(image &&rhs) = default;

  ~image() = default;

  template <typename dataT, access::mode accessmode>
  accessor<dataT, dimensions, accessmode,
           access::target::image, access::placeholder::false_t>
  get_access(handler &commandGroupHandler) {
    return accessor<dataT, dimensions, accessmode, access::target::image, access::placeholder::false_t>{};
  }

  template <typename dataT, access::mode accessmode>
  accessor<dataT, dimensions, accessmode,
           access::target::host_image, access::placeholder::false_t>
  get_access() {
    return accessor<dataT, dimensions, accessmode, access::target::host_image, access::placeholder::false_t>{};
  }
};

} // namespace _V1
} // namespace sycl
