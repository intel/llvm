//==----- memcpy2d.hpp --- SYCL 2D memcpy extension ------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <sycl/handler.hpp>
#include <sycl/queue.hpp>
#include <sycl/usm/usm_enums.hpp>
#include <sycl/usm/usm_pointer_info.hpp>

namespace sycl {
inline namespace _V1 {
template <typename T, typename>
void handler::ext_oneapi_memcpy2d(void *Dest, size_t DestPitch, const void *Src,
                                  size_t SrcPitch, size_t Width,
                                  size_t Height) {
  throwIfGraphAssociated<
      ext::oneapi::experimental::detail::UnsupportedGraphFeatures::
          sycl_ext_oneapi_memcpy2d>();
  throwIfActionIsCreated();
  if (Width > DestPitch)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Destination pitch must be greater than or equal "
                          "to the width specified in 'ext_oneapi_memcpy2d'");
  if (Width > SrcPitch)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Source pitch must be greater than or equal "
                          "to the width specified in 'ext_oneapi_memcpy2d'");

  // Get the type of the pointers.
  context Ctx = detail::createSyclObjFromImpl<context>(getContextImplPtr());
  usm::alloc SrcAllocType = get_pointer_type(Src, Ctx);
  usm::alloc DestAllocType = get_pointer_type(Dest, Ctx);
  bool SrcIsHost =
      SrcAllocType == usm::alloc::unknown || SrcAllocType == usm::alloc::host;
  bool DestIsHost =
      DestAllocType == usm::alloc::unknown || DestAllocType == usm::alloc::host;

  // Do the following:
  // 1. If both are host, use host_task to copy.
  // 2. If either pointer is host or the backend supports native memcpy2d, use
  //    special command.
  // 3. Otherwise, launch a kernel for copying.
  if (SrcIsHost && DestIsHost) {
    commonUSMCopy2DFallbackHostTask<T>(Src, SrcPitch, Dest, DestPitch, Width,
                                       Height);
  } else if (SrcIsHost || DestIsHost || supportsUSMMemcpy2D()) {
    ext_oneapi_memcpy2d_impl(Dest, DestPitch, Src, SrcPitch, Width, Height);
  } else {
    commonUSMCopy2DFallbackKernel<T>(Src, SrcPitch, Dest, DestPitch, Width,
                                     Height);
  }
}

template <typename T>
void handler::ext_oneapi_copy2d(const T *Src, size_t SrcPitch, T *Dest,
                                size_t DestPitch, size_t Width, size_t Height) {
  if (Width > DestPitch)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Destination pitch must be greater than or equal "
                          "to the width specified in 'ext_oneapi_copy2d'");
  if (Width > SrcPitch)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Source pitch must be greater than or equal "
                          "to the width specified in 'ext_oneapi_copy2d'");

  // Get the type of the pointers.
  context Ctx = detail::createSyclObjFromImpl<context>(getContextImplPtr());
  usm::alloc SrcAllocType = get_pointer_type(Src, Ctx);
  usm::alloc DestAllocType = get_pointer_type(Dest, Ctx);
  bool SrcIsHost =
      SrcAllocType == usm::alloc::unknown || SrcAllocType == usm::alloc::host;
  bool DestIsHost =
      DestAllocType == usm::alloc::unknown || DestAllocType == usm::alloc::host;

  // Do the following:
  // 1. If both are host, use host_task to copy.
  // 2. If either pointer is host or of the backend supports native memcpy2d,
  //    use special command.
  // 3. Otherwise, launch a kernel for copying.
  if (SrcIsHost && DestIsHost) {
    commonUSMCopy2DFallbackHostTask<T>(Src, SrcPitch, Dest, DestPitch, Width,
                                       Height);
  } else if (SrcIsHost || DestIsHost || supportsUSMMemcpy2D()) {
    ext_oneapi_memcpy2d_impl(Dest, DestPitch * sizeof(T), Src,
                             SrcPitch * sizeof(T), Width * sizeof(T), Height);
  } else {
    commonUSMCopy2DFallbackKernel<T>(Src, SrcPitch, Dest, DestPitch, Width,
                                     Height);
  }
}

template <typename T, typename>
void handler::ext_oneapi_memset2d(void *Dest, size_t DestPitch, int Value,
                                  size_t Width, size_t Height) {
  throwIfActionIsCreated();
  if (Width > DestPitch)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Destination pitch must be greater than or equal "
                          "to the width specified in 'ext_oneapi_memset2d'");
  T CharVal = static_cast<T>(Value);

  context Ctx = detail::createSyclObjFromImpl<context>(getContextImplPtr());
  usm::alloc DestAllocType = get_pointer_type(Dest, Ctx);

  // If the backends supports 2D fill we use that. Otherwise we use a fallback
  // kernel. If the target is on host we will always do the operation on host.
  if (DestAllocType == usm::alloc::unknown || DestAllocType == usm::alloc::host)
    commonUSMFill2DFallbackHostTask(Dest, DestPitch, CharVal, Width, Height);
  else if (supportsUSMMemset2D())
    ext_oneapi_memset2d_impl(Dest, DestPitch, Value, Width, Height);
  else
    commonUSMFill2DFallbackKernel(Dest, DestPitch, CharVal, Width, Height);
}

template <typename T>
void handler::ext_oneapi_fill2d(void *Dest, size_t DestPitch, const T &Pattern,
                                size_t Width, size_t Height) {
  throwIfActionIsCreated();
  static_assert(is_device_copyable<T>::value,
                "Pattern must be device copyable");
  if (Width > DestPitch)
    throw sycl::exception(sycl::make_error_code(errc::invalid),
                          "Destination pitch must be greater than or equal "
                          "to the width specified in 'ext_oneapi_fill2d'");

  context Ctx = detail::createSyclObjFromImpl<context>(getContextImplPtr());
  usm::alloc DestAllocType = get_pointer_type(Dest, Ctx);

  // If the backends supports 2D fill we use that. Otherwise we use a fallback
  // kernel. If the target is on host we will always do the operation on host.
  if (DestAllocType == usm::alloc::unknown || DestAllocType == usm::alloc::host)
    commonUSMFill2DFallbackHostTask(Dest, DestPitch, Pattern, Width, Height);
  else if (supportsUSMFill2D())
    ext_oneapi_fill2d_impl(Dest, DestPitch, &Pattern, sizeof(T), Width, Height);
  else
    commonUSMFill2DFallbackKernel(Dest, DestPitch, Pattern, Width, Height);
}

template <typename T, typename>
event queue::ext_oneapi_memcpy2d(void *Dest, size_t DestPitch, const void *Src,
                                 size_t SrcPitch, size_t Width, size_t Height,
                                 event DepEvent,
                                 const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_memcpy2d<T>(Dest, DestPitch, Src, SrcPitch, Width,
                                   Height);
      },
      CodeLoc);
}

template <typename T, typename>
event queue::ext_oneapi_memcpy2d(void *Dest, size_t DestPitch, const void *Src,
                                 size_t SrcPitch, size_t Width, size_t Height,
                                 const std::vector<event> &DepEvents,
                                 const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_memcpy2d<T>(Dest, DestPitch, Src, SrcPitch, Width,
                                   Height);
      },
      CodeLoc);
}

template <typename T>
event queue::ext_oneapi_copy2d(const T *Src, size_t SrcPitch, T *Dest,
                               size_t DestPitch, size_t Width, size_t Height,
                               const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.ext_oneapi_copy2d<T>(Src, SrcPitch, Dest, DestPitch, Width, Height);
      },
      CodeLoc);
}

template <typename T>
event queue::ext_oneapi_copy2d(const T *Src, size_t SrcPitch, T *Dest,
                               size_t DestPitch, size_t Width, size_t Height,
                               event DepEvent,
                               const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_copy2d<T>(Src, SrcPitch, Dest, DestPitch, Width, Height);
      },
      CodeLoc);
}

template <typename T>
event queue::ext_oneapi_copy2d(const T *Src, size_t SrcPitch, T *Dest,
                               size_t DestPitch, size_t Width, size_t Height,
                               const std::vector<event> &DepEvents,
                               const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_copy2d<T>(Src, SrcPitch, Dest, DestPitch, Width, Height);
      },
      CodeLoc);
}

template <typename T, typename>
event queue::ext_oneapi_memset2d(void *Dest, size_t DestPitch, int Value,
                                 size_t Width, size_t Height,
                                 const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.ext_oneapi_memset2d<T>(Dest, DestPitch, Value, Width, Height);
      },
      CodeLoc);
}

template <typename T, typename>
event queue::ext_oneapi_memset2d(void *Dest, size_t DestPitch, int Value,
                                 size_t Width, size_t Height, event DepEvent,
                                 const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_memset2d<T>(Dest, DestPitch, Value, Width, Height);
      },
      CodeLoc);
}

template <typename T, typename>
event queue::ext_oneapi_memset2d(void *Dest, size_t DestPitch, int Value,
                                 size_t Width, size_t Height,
                                 const std::vector<event> &DepEvents,
                                 const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_memset2d<T>(Dest, DestPitch, Value, Width, Height);
      },
      CodeLoc);
}

template <typename T>
event queue::ext_oneapi_fill2d(void *Dest, size_t DestPitch, const T &Pattern,
                               size_t Width, size_t Height,
                               const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.ext_oneapi_fill2d<T>(Dest, DestPitch, Pattern, Width, Height);
      },
      CodeLoc);
}

template <typename T>
event queue::ext_oneapi_fill2d(void *Dest, size_t DestPitch, const T &Pattern,
                               size_t Width, size_t Height, event DepEvent,
                               const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvent);
        CGH.ext_oneapi_fill2d<T>(Dest, DestPitch, Pattern, Width, Height);
      },
      CodeLoc);
}

template <typename T>
event queue::ext_oneapi_fill2d(void *Dest, size_t DestPitch, const T &Pattern,
                               size_t Width, size_t Height,
                               const std::vector<event> &DepEvents,
                               const detail::code_location &CodeLoc) {
  return submit(
      [=](handler &CGH) {
        CGH.depends_on(DepEvents);
        CGH.ext_oneapi_fill2d<T>(Dest, DestPitch, Pattern, Width, Height);
      },
      CodeLoc);
}
} // namespace _V1
} // namespace sycl
