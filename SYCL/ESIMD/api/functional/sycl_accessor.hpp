//===-- sycl_accessor.hpp - Test coverasge support for sycl::accessor -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides type packs and all neccessary helpers and wrappers to use with
/// tests with sycl::accessor
///
//===----------------------------------------------------------------------===//

#pragma once

#include "logger.hpp"
#include "type_coverage.hpp"

#include <sstream>

namespace esimd_test::api::functional {

//----------------------------------------------------------------------------//
// Type packs and helpers for accessor dimensions
//----------------------------------------------------------------------------//

enum class accessor_dimension_group { all = 0, non_zero, zero };

// It's UB to access the data by using zero-sized accessor, still 0 dimensions
// is valid value in SYCL 2020.
// See https://github.com/KhronosGroup/SYCL-Docs/pull/192 for details
//
// It's expected that any /p group value axcept `non_zero` is a corner case for
// now
template <accessor_dimension_group group = accessor_dimension_group::non_zero>
inline auto get_accessor_dimensions() {
  if constexpr (group == accessor_dimension_group::all) {
    return integer_pack<0, 1, 2, 3>::generate_unnamed();
  } else if constexpr (group == accessor_dimension_group::non_zero) {
    return integer_pack<1, 2, 3>::generate_unnamed();
  } else if constexpr (group == accessor_dimension_group::zero) {
    return integer_pack<0>::generate_unnamed();
  } else {
    static_assert(group != group, "Unexpected accessor dimensions group");
  }
}

//----------------------------------------------------------------------------//
// Type packs and helpers for accessor modes
//----------------------------------------------------------------------------//

// Alias to use for accessor modes; no overhead as alias doesn't declare a new
// type
template <sycl::access_mode... values>
using accessor_mode_pack = value_pack<sycl::access_mode, values...>;

enum class accessor_mode_group {
  all = 0,     // all possible values
  all_read,    // all possible values for read access
  single_read, // single value from the all_read group
  all_write,
  single_write,
  all_read_write,
  single_read_write = all_read_write
};

namespace log {
// Specialization of generic stringification for accessor mode logging purposes
template <> struct StringMaker<sycl::access_mode> {
  static std::string stringify(sycl::access_mode mode) {
    std::string result;
    switch (mode) {
    case sycl::access_mode::read:
      result = "read";
      break;
    case sycl::access_mode::write:
      result = "write";
      break;
    case sycl::access_mode::read_write:
      result = "read_write";
      break;
    default:
      result = "unknown";
    };
    return result;
  }
};
} // namespace log

template <accessor_mode_group group> class accessor_modes {
  static auto pack_type_helper() {
    using ModeT = sycl::access_mode;

    if constexpr (group == accessor_mode_group::all) {
      using ModesT =
          accessor_mode_pack<ModeT::read, ModeT::write, ModeT::read_write>;
      return ModesT{};
    } else if constexpr (group == accessor_mode_group::all_read) {
      using ModesT = accessor_mode_pack<ModeT::read, ModeT::read_write>;
      return ModesT{};
    } else if constexpr (group == accessor_mode_group::single_read) {
      using ModesT = accessor_mode_pack<ModeT::read>;
      return ModesT{};
    } else if constexpr (group == accessor_mode_group::all_write) {
      using ModesT = accessor_mode_pack<ModeT::write, ModeT::read_write>;
      return ModesT{};
    } else if constexpr (group == accessor_mode_group::single_write) {
      using ModesT = accessor_mode_pack<ModeT::write>;
      return ModesT{};
    } else if constexpr (group == accessor_mode_group::all_read_write) {
      using ModesT = accessor_mode_pack<ModeT::read_write>;
      return ModesT{};
    } else {
      static_assert(group != group, "Unexpected accessor mode group");
    }
  }

public:
  static auto generate_unnamed() {
    return pack_type_helper().generate_unnamed();
  }

  static auto generate_named() {
    static const auto generator = &log::stringify<sycl::access_mode>;
    return pack_type_helper().generate_named_by(generator);
  }
};

//----------------------------------------------------------------------------//
// Type packs and helpers for accessor targets
//----------------------------------------------------------------------------//

// Alias to use for accessor targets; no overhead as alias doesn't declare a new
// type
template <sycl::target... values>
using accessor_target_pack = value_pack<sycl::target, values...>;

namespace log {
// Specialization of generic stringification for accessor target logging
// purposes
template <> struct StringMaker<sycl::target> {
  static std::string stringify(sycl::target target) {
    std::string result;
    switch (target) {
    case sycl::target::device:
      result = "device";
      break;
    default:
      result = "unknown";
    };
    return result;
  }
};
} // namespace log

// Makes easy to add new accessor targets support
class accessor_targets {
  using pack_type = accessor_target_pack<sycl::target::device>;

public:
  static auto generate_unnamed() { return pack_type::generate_unnamed(); }

  static auto generate_named() {
    return pack_type::generate_named_by(log::stringify<sycl::target>);
  }
};

//----------------------------------------------------------------------------//
// Type packs and helpers for accessor type itself
//----------------------------------------------------------------------------//

// Provides generic accessor type stringification
struct AccessorDescription : public ITestDescription {
public:
  // Parameters are given in the order they are printed
  //
  // Usage examples:
  //
  //   // Declare "sycl::accessor<int, 1, read_write>"
  //   const auto foo = AccessorDescription("int", 1, "read_write");
  //
  //   // Declare "sycl::accessor<int, 1, read, device>"
  //   const auto mode = sycl::access_mode::read;
  //   const auto target = sycl::target::device;
  //   const auto bar = AccessorDescription("int", "1", mode, target);
  //
  template <typename... ArgsT>
  AccessorDescription(const std::string &data_type_name, ArgsT &&...args) {
    std::ostringstream stream;
    stream << "sycl::accessor<" << data_type_name;
    ((stream << ", " << log::stringify(std::forward<ArgsT>(args))), ...);
    stream << ">";
    m_description = stream.str();
  }

  std::string to_string() const override { return m_description; }

private:
  std::string m_description;
};

} // namespace esimd_test::api::functional
