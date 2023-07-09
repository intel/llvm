#include <sycl/detail/usm_impl.hpp>
#include <sycl/ext/oneapi/annotated_arg/annotated_ptr.hpp>
#include <sycl/sycl.hpp>

namespace sycl {
__SYCL_INLINE_VER_NAMESPACE(_V1) {
namespace ext {
namespace oneapi {
namespace experimental {

using alloc = sycl::usm::alloc;

namespace {

// Transform a compile-time property list to a USM property_list (working at
// runtime). Right now only the `buffer_location<N>` has its corresponding USM
// runtime property and is transformable
template <typename... Props>
typename std::enable_if_t<detail::properties_t<Props...>::template has_property<
                              buffer_location_key>(),
                          property_list>
get_usm_property_list(const detail::properties_t<Props...> &propList) {
  using property_list_t = detail::properties_t<Props...>;
  auto bufferLocationProp =
      property_list_t::template get_property<buffer_location_key>();
  int BufferLocationId =
      detail::PropertyMetaInfo<decltype(bufferLocationProp)>::value;
  return property_list{ext::intel::experimental::property::usm::buffer_location(
      BufferLocationId)};
}

// Return empty propert_list when compile-time property buffer_location is not
// present in the given property list
template <typename... Props>
typename std::enable_if_t<detail::properties_t<Props...>::template has_property<
                              buffer_location_key>() == false,
                          property_list>
get_usm_property_list(const detail::properties_t<Props...> &propList) {
  return {};
}

// Get the value of compile-time property `alignment` in a given property list.
template <typename... Props>
typename std::enable_if_t<
    detail::properties_t<Props...>::template has_property<alignment_key>(),
    size_t>
get_align_from_property_list(const detail::properties_t<Props...> &propList) {
  using property_list_t = detail::properties_t<Props...>;
  using align_val_type =
      decltype(property_list_t::template get_property<alignment_key>());
  return detail::PropertyMetaInfo<align_val_type>::value;
}

// Return 0 if alignment is not present in a given compile-time property list
template <typename... Props>
typename std::enable_if_t<detail::properties_t<Props...>::template has_property<
                              alignment_key>() == false,
                          size_t>
get_align_from_property_list(const detail::properties_t<Props...> &propList) {
  return 0;
}

// Get the value of compile-time property `usm_kind` in a given property list.
template <typename... Props>
typename std::enable_if_t<
    detail::properties_t<Props...>::template has_property<usm_key>(), alloc>
get_usm_kind_from_property_list(
    const detail::properties_t<Props...> &propList) {
  using property_list_t = detail::properties_t<Props...>;
  using usm_kind_val_type =
      decltype(property_list_t::template get_property<usm_key>());
  return detail::PropertyMetaInfo<usm_kind_val_type>::value;
}

// Return 0 if usm_kind is not present in a given compile-time property list
template <typename... Props>
typename std::enable_if_t<
    detail::properties_t<Props...>::template has_property<usm_key>() == false,
    size_t>
get_usm_kind_from_property_list(
    const detail::properties_t<Props...> &propList) {
  return alloc::unknown;
}

size_t combine_align(size_t alignA, size_t alignB) {
  if (!alignA)
    return alignB;
  else if (!alignB)
    return alignA;
  return std::lcm(alignA, alignB);
}

} // anonymous namespace

// Filter the compile-time properties from a property list
template <typename PropertyListT> struct FilterProperties {
  static_assert(is_property_list<PropertyListT>::value,
                "Property list is invalid.");
};

// Partial specialization for the property filter
template <typename Prop> struct FilterProperties<detail::properties_t<Prop>> {
  using type =
      std::conditional_t<detail::IsCompileTimePropertyValue<Prop>::value,
                         detail::properties_t<Prop>,
                         detail::empty_properties_t>;
};

// Partial specialization for the property filter
template <typename Prop, typename... Props>
struct FilterProperties<detail::properties_t<Prop, Props...>> {
  using filtered_this_property_t =
      std::conditional_t<detail::IsCompileTimePropertyValue<Prop>::value,
                         detail::properties_t<Prop>,
                         detail::empty_properties_t>;
  using filtered_other_properties_t =
      FilterProperties<detail::properties_t<Props...>>::type;
  using type = detail::merged_properties_t<filtered_this_property_t,
                                           filtered_other_properties_t>;
};

// Generate the property list for the annotated_ptr output of annotated USM
// alloc APIs
template <sycl::usm::alloc Kind, typename PropertyListT>
struct GetAnnotatedPtrProperties {};

// Partial specialization
template <sycl::usm::alloc Kind, typename... Props>
struct GetAnnotatedPtrProperties<Kind, detail::properties_t<Props...>> {
  using input_properties_t = detail::properties_t<Props...>;
  using filtered_input_properties_t =
      FilterProperties<input_properties_t>::type;

  using type =
      detail::merged_properties_t<filtered_input_properties_t,
                                  decltype(properties{usm_kind<Kind>})>;
};

////
//  Device USM allocation functions with properties support
////

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_device_annotated(size_t numBytes, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {
  size_t align = get_align_from_property_list(propList);
  return aligned_alloc_device_annotated<propertyListA, propertyListB>(
      align, numBytes, syclDevice, syclContext, propList);
}

template <typename propertyListA, typename propertyListB,
          typename = std::enable_if_t<std::is_same<
              propertyListB, typename GetAnnotatedPtrProperties<
                                 alloc::device, propertyListA>::type>::value>>
auto aligned_alloc_device_annotated(
    size_t alignment, size_t numBytes, const device &syclDevice,
    const context &syclContext, const propertyListA &propList = properties{}) {

  static_assert(get_usm_kind_from_property_list(propList) == alloc::device &&
                "USM kind in the property list is not 'usm_kind::device' when "
                "allocating device memory");

  size_t alignFromPropList = get_align_from_property_list(propList);
  const property_list &usmPropList = get_usm_property_list(propList);
  void *rawPtr = sycl::aligned_alloc_device(
      combine_align(alignment, alignFromPropList), numBytes, syclDevice,
      syclContext, usmPropList);
  return annotated_ptr<void, propertyListB>(rawPtr);
}

////
//  Host USM allocation functions with properties support
//
////

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_host_annotated(size_t numBytes, const context &syclContext,
                      const propertyListA &propList = properties{}) {

  size_t align = get_align_from_property_list(propList);
  return aligned_alloc_host_annotated<propertyListA, propertyListB>(
      align, numBytes, syclContext, propList);
}

template <typename propertyListA, typename propertyListB,
          typename = std::enable_if_t<std::is_same<
              propertyListB, typename GetAnnotatedPtrProperties<
                                 alloc::host, propertyListA>::type>::value>>
auto aligned_alloc_host_annotated(
    size_t alignment, size_t numBytes, const context &syclContext,
    const propertyListA &propList = properties{}) {

  static_assert(get_usm_kind_from_property_list(propList) == alloc::host &&
                "USM kind in the property list is not 'usm_kind::host' when "
                "allocating host memory");

  size_t alignFromPropList = get_align_from_property_list(propList);
  const property_list &usmPropList = get_usm_property_list(propList);
  void *rawPtr =
      sycl::aligned_alloc_host(combine_align(alignment, alignFromPropList),
                               numBytes, syclContext, usmPropList);
  return annotated_ptr<void, propertyListB>(rawPtr);
}

////
//  Shared USM allocation functions with properties support
////
template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_shared_annotated(size_t numBytes, const device &syclDevice,
                        const context &syclContext,
                        const propertyListA &propList = properties{}) {

  size_t align = get_align_from_property_list(propList);
  return aligned_alloc_shared_annotated<propertyListA, propertyListB>(
      align, numBytes, syclDevice, syclContext, propList);
}

template <typename propertyListA, typename propertyListB,
          typename = std::enable_if_t<std::is_same<
              propertyListB, typename GetAnnotatedPtrProperties<
                                 alloc::shared, propertyListA>::type>::value>>
auto aligned_alloc_shared_annotated(
    size_t alignment, size_t numBytes, const device &syclDevice,
    const context &syclContext, const propertyListA &propList = properties{}) {

  static_assert(get_usm_kind_from_property_list(propList) == alloc::shared &&
                "USM kind in the property list is not 'usm_kind::shared' when "
                "allocating shared memory");

  size_t alignFromPropList = get_align_from_property_list(propList);
  const property_list &usmPropList = get_usm_property_list(propList);
  void *rawPtr = sycl::aligned_alloc_shared(
      combine_align(alignment, alignFromPropList), numBytes, syclDevice,
      syclContext, usmPropList);
  return annotated_ptr<void, propertyListB>(rawPtr);
}

////
//  Parameterized USM allocation functions with properties support
////
template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, sycl::usm::alloc kind,
                 const propertyListA &propList = properties{}) {
  size_t align = get_align_from_property_list(propList);
  return aligned_alloc_annotated<propertyListA, propertyListB>(
      align, numBytes, syclDevice, syclContext, kind, propList);
}

template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
aligned_alloc_annotated(size_t alignment, size_t numBytes,
                        const device &syclDevice, const context &syclContext,
                        sycl::usm::alloc kind,
                        const propertyListA &propList = properties{}) {

  if (kind == get_usm_kind_from_property_list(propList)) {
    throw sycl::exception(make_error_code(errc::invalid),
                          "USM kind argument conflicts with property list");
  }
  switch (kind) {
  case device:
    return aligned_alloc_device_annotated<propertyListA, propertyListB>(
        alignment, numBytes, syclDevice, syclContext, propList);
  case host:
    return aligned_alloc_host_annotated<propertyListA, propertyListB>(
        alignment, numBytes, syclContext, propList);
  default:
    return aligned_alloc_shared_annotated<propertyListA, propertyListB>(
        alignment, numBytes, syclDevice, syclContext, propList);
  }
}

////
//  Additional USM memory allocation functions, requiring the usm_kind property
////
template <typename propertyListA, typename propertyListB>
annotated_ptr<void, propertyListB>
malloc_annotated(size_t numBytes, const device &syclDevice,
                 const context &syclContext, const propertyListA &propList) {

  alloc kind = get_usm_kind_from_property_list(propList);
  return malloc_annotated<propertyListA, propertyListB>(
      numBytes, syclDevice, syclContext, kind, propList);
}

} // namespace experimental
} // namespace oneapi
} // namespace ext
} // __SYCL_INLINE_VER_NAMESPACE(_V1)
} // namespace sycl