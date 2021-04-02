//==------- kernel_bundle.cpp - SYCL kernel_bundle and free functions ------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/kernel_bundle_impl.hpp>
#include <detail/kernel_id_impl.hpp>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

kernel_id::kernel_id(const char *Name)
    : impl(std::make_shared<detail::kernel_id_impl>(Name)) {}

const char *kernel_id::get_name() const noexcept { return impl->get_name(); }

namespace detail {

////////////////////////////
/////  device_image_plain
///////////////////////////

bool device_image_plain::has_kernel(const kernel_id &KernelID) const noexcept {
  return impl->has_kernel(KernelID);
}

bool device_image_plain::has_kernel(const kernel_id &KernelID,
                                    const device &Dev) const noexcept {
  return impl->has_kernel(KernelID, Dev);
}

////////////////////////////
///// kernel_bundle_plain
///////////////////////////

bool kernel_bundle_plain::empty() const noexcept { return impl->empty(); }

backend kernel_bundle_plain::get_backend() const noexcept {
  return impl->get_backend();
}

context kernel_bundle_plain::get_context() const noexcept {
  return impl->get_context();
}

std::vector<device> kernel_bundle_plain::get_devices() const noexcept {
  return impl->get_devices();
}

std::vector<kernel_id> kernel_bundle_plain::get_kernel_ids() const {
  return impl->get_kernel_ids();
}

bool kernel_bundle_plain::contains_specialization_constants() const noexcept {
  return impl->contains_specialization_constants();
}

bool kernel_bundle_plain::native_specialization_constant() const noexcept {
  return impl->native_specialization_constant();
}

kernel kernel_bundle_plain::get_kernel(const kernel_id &KernelID) const {
  return impl->get_kernel(KernelID, impl);
}

const device_image_plain *kernel_bundle_plain::begin() const {
  return impl->begin();
}

const device_image_plain *kernel_bundle_plain::end() const {
  return impl->end();
}

bool kernel_bundle_plain::has_kernel(const kernel_id &KernelID) const noexcept {
  return impl->has_kernel(KernelID);
}

bool kernel_bundle_plain::has_kernel(const kernel_id &KernelID,
                                     const device &Dev) const noexcept {
  return impl->has_kernel(KernelID, Dev);
}

bool kernel_bundle_plain::has_specialization_constant_impl(
    const char *SpecName) const noexcept {
  return impl->has_specialization_constant(SpecName);
}

void kernel_bundle_plain::set_specialization_constant_impl(const char *SpecName,
                                                           void *Value,
                                                           size_t Size) {
  impl->set_specialization_constant_raw_value(SpecName, Value, Size);
}

void kernel_bundle_plain::get_specialization_constant_impl(const char *SpecName,
                                                           void *Value,
                                                           size_t Size) const {
  impl->get_specialization_constant_raw_value(SpecName, Value, Size);
}

////////////////////////////
///// free functions
///////////////////////////

detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       bundle_state State) {
  return std::make_shared<detail::kernel_bundle_impl>(Ctx, Devs, State);
}

detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       const std::vector<kernel_id> &KernelIDs,
                       bundle_state State) {
  return std::make_shared<detail::kernel_bundle_impl>(Ctx, Devs, KernelIDs,
                                                      State);
}

detail::KernelBundleImplPtr
get_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                       bundle_state State, const DevImgSelectorImpl &Selector) {
  return std::make_shared<detail::kernel_bundle_impl>(Ctx, Devs, Selector,
                                                      State);
}

std::shared_ptr<detail::kernel_bundle_impl>
join_impl(const std::vector<detail::KernelBundleImplPtr> &Bundles) {
  return std::make_shared<detail::kernel_bundle_impl>(Bundles);
}

bool has_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                            bundle_state State) {
  // Just create a kernel_bundle and check if it has any device_images inside.
  detail::kernel_bundle_impl KernelBundleImpl(Ctx, Devs, State);
  return KernelBundleImpl.size();
}

bool has_kernel_bundle_impl(const context &Ctx, const std::vector<device> &Devs,
                            const std::vector<kernel_id> &KernelIds,
                            bundle_state State) {
  // Just create a kernel_bundle and check if it has any device_images inside.
  detail::kernel_bundle_impl KernelBundleImpl(Ctx, Devs, KernelIds, State);
  return KernelBundleImpl.size();
}

std::shared_ptr<detail::kernel_bundle_impl>
compile_impl(const kernel_bundle<bundle_state::input> &InputBundle,
             const std::vector<device> &Devs, const property_list &PropList) {
  return std::make_shared<detail::kernel_bundle_impl>(
      InputBundle, Devs, PropList, bundle_state::object);
}

std::shared_ptr<detail::kernel_bundle_impl>
link_impl(const std::vector<kernel_bundle<bundle_state::object>> &ObjectBundles,
          const std::vector<device> &Devs, const property_list &PropList) {
  return std::make_shared<detail::kernel_bundle_impl>(ObjectBundles, Devs,
                                                      PropList);
}

std::shared_ptr<detail::kernel_bundle_impl>
build_impl(const kernel_bundle<bundle_state::input> &InputBundle,
           const std::vector<device> &Devs, const property_list &PropList) {
  return std::make_shared<detail::kernel_bundle_impl>(
      InputBundle, Devs, PropList, bundle_state::executable);
}

} // namespace detail

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
