#pragma once

#define ATTR_SYCL_KERNEL __attribute__((sycl_kernel))

// Dummy runtime classes to model SYCL API.
namespace cl {
namespace sycl {
struct sampler_impl {
#ifdef __SYCL_DEVICE_ONLY__
  __ocl_sampler_t m_Sampler;
#endif
};

class sampler {
  struct sampler_impl impl;
#ifdef __SYCL_DEVICE_ONLY__
  void __init(__ocl_sampler_t Sampler) { impl.m_Sampler = Sampler; }
#endif

public:
  void use(void) const {}
};

template <int dimensions = 1>
class group {
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
  template <typename... propertyTN>
  property_list(propertyTN... props) {}

  template <typename propertyT>
  bool has_property() const { return true; }

  template <typename propertyT>
  propertyT get_property() const {
    return propertyT{};
  }

  bool operator==(const property_list &rhs) const { return false; }

  bool operator!=(const property_list &rhs) const { return false; }
};

template <int dim>
struct id {
  template <typename... T>
  id(T... args) {} // fake constructor
private:
  // Some fake field added to see using of id arguments in the
  // kernel wrapper
  int Data;
};

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
          access::placeholder isPlaceholder = access::placeholder::false_t>
class accessor {

public:
  void use(void) const {}
  template <typename... T>
  void use(T... args) {}
  template <typename... T>
  void use(T... args) const {}
  _ImplT<dimensions> impl;

private:
  void __init(__attribute__((ocl_global)) dataT *Ptr, range<dimensions> AccessRange,
              range<dimensions> MemRange, id<dimensions> Offset) {}
};

template <int dimensions, access::mode accessmode, access::target accesstarget>
struct opencl_image_type;

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
class accessor<dataT, dimensions, accessmode, access::target::image, access::placeholder::false_t> {
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

class handler {
public:
  template <typename KernelName, typename KernelType, int dimensions>
  ATTR_SYCL_KERNEL
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc) {}

  template <typename KernelName, typename KernelType, int dimensions>
  ATTR_SYCL_KERNEL
  void parallel_for(nd_range<dimensions> executionRange,
                    KernelType kernelFunc) {}

  template <int dimensions>
  ATTR_SYCL_KERNEL
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel) {}

  template <int dimensions>
  ATTR_SYCL_KERNEL
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel) {}

  template <typename KernelName, typename KernelType>
  ATTR_SYCL_KERNEL
  void single_task(KernelType kernelFunc) {}

  template <typename KernelType>
  ATTR_SYCL_KERNEL
  void single_task(KernelType kernelFunc) {}

  template <typename KernelName, typename KernelType, int dimensions>
  ATTR_SYCL_KERNEL
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel,
                    KernelType kernelFunc) {}

  template <typename KernelType, int dimensions>
  ATTR_SYCL_KERNEL
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc) {}

  template <typename KernelName, typename KernelType, int dimensions>
  ATTR_SYCL_KERNEL
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel,
                    KernelType kernelFunc) {}
};

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
        const range<dimensions> &Range, const property_list &PropList = {}) {}

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
