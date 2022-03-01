#pragma once

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))

extern "C" int printf(const char* fmt, ...);

// Dummy runtime classes to model SYCL API.
inline namespace cl {
namespace sycl {
struct sampler_impl {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_sampler_t m_Sampler;
#endif
};

class __attribute__((sycl_special_class)) sampler {
  struct sampler_impl impl;
#ifdef __SYCL_DEVICE_ONLY__
  void __init(__ocl_sampler_t Sampler) { impl.m_Sampler = Sampler; }
#endif

public:
  void use(void) const {}
};

template <int dimensions = 1>
class group {
public:
  group() = default; // fake constructor
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
  local_space
};
} // namespace access

// Dummy aspect enum with limited enumerators
enum class aspect {
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
struct buffer_location {
  template <int> class instance {};
};
} // namespace property
} // namespace intel
} // namespace ext

namespace ext {
namespace oneapi {
namespace property {
// Compile time known accessor property
struct no_alias {
  template <bool> class instance {};
};
} // namespace property

// device_global type decorated with attributes
template <typename T>
class [[__sycl_detail__::device_global]]
[[__sycl_detail__::global_variable_allowed]] device_global {
public :
  const T & get() const noexcept { return *Data; }
  device_global() {}
  operator T&() noexcept { return *Data; }
private:
  T *Data;
};

// decorated with only global_variable_allowed attribute
template <typename T>
class [[__sycl_detail__::global_variable_allowed]] only_global_var_allowed {
public :
  const T & get() const noexcept { return *Data; }
  only_global_var_allowed() {}
  operator T&() noexcept { return *Data; }
private:
  T *Data;
};
} // namespace oneapi
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
class accessor_property_list {};
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
class __attribute__((sycl_special_class)) accessor {

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
class __attribute__((sycl_special_class)) accessor<dataT, dimensions, accessmode, access::target::image, access::placeholder::false_t> {
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
template <typename T, typename ID = T>
class spec_constant {
public:
  spec_constant() {}
  spec_constant(T Cst) {}

  T get() const { // explicit access.
    return T();   // Dummy implementaion.
  }
  operator T() const { // implicit conversion.
    return get();
  }
};

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

} // namespace experimental
} // namespace oneapi
} // namespace ext

class kernel_handler {
  void __init_specialization_constants_buffer(char *specialization_constants_buffer) {}
};

template <typename T> class specialization_id {
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

template <typename KernelName, typename KernelType, int Dims>
ATTR_SYCL_KERNEL void
kernel_parallel_for_work_group(const KernelType &KernelFunc) {
  KernelFunc(group<Dims>());
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

class __attribute__((sycl_special_class)) stream {
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
  cl::sycl::accessor<char, 1, cl::sycl::access::mode::read_write> Acc;
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

} // namespace sycl
} // namespace cl
