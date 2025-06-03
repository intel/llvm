# Helper utility to generate sycl/version.hpp file which contains various
# identifiers which can be used to distinguish different builds between each
# other:
#
# __LIBSYCL_TIMESTAMP - timestamp of the latest commit made into sycl/ directory
# in YYYYMMDD format
#
# __SYCL_COMPILER_VERSION - date when configure step was launched in YYYYMMDD
# format. Deprecated
#
# Note: it may not always be the case that CMake configuration step was re-run
# when a new commits is added and therefore, execute_process won't be re-run and
# the hash won't be updated. It is not considered to be a problem, because for
# local incremental builds made during the library/headers development the date
# doesn't matter much and we can guarantee it to be always correct when we do
# nightly builds.

# Grab the date of the latest commit in sycl folder
execute_process(
  # date in YYYYMMDD mode, see strftime for reference
  COMMAND git log -1 --format=%ad --date=format:%Y%m%d -- ${SYCL_ROOT_SOURCE_DIR}
  OUTPUT_VARIABLE __LIBSYCL_TIMESTAMP
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Legacy thing for backwards compatibility. Use of the current date is not
# reliable, because we can always make new build from older commits.
string(TIMESTAMP __SYCL_COMPILER_VERSION "%Y%m%d")
configure_file(
  "${sycl_src_dir}/version.hpp.in"
  "${SYCL_INCLUDE_BUILD_DIR}/sycl/version.hpp"
)
