# Freeze base image to avoid using newer GPU RT/IGC when updating 
# the image due to changes in the release branch.
ARG base_tag=1dcf3fd93f3c13b21e9931372fe405493f7e5a7bd5ba1fca192dece74f9b9188
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_intel_drivers

# Remove @sha256 if replacing hash with label in base_tag
FROM $base_image@sha256:$base_tag

# Actual CI maps volumes via something like "/runner/host/path":"/__w/", so
# these won't be visible there. They can be used when manually reproducing
# issues though. Path `/__w/llvm/llvm` is the property of Github Actions and
# when CMake configures E2E tests this path is hardcoded in both CMake files and
# e2e binaries themselve. As such, it's useful to have these in the image for
# local manual experiments.
#
# One can map volumes as "/host/system/new/toolchain":"/__w/llvm/llvm/toolchain"
# to override the toolchain in order to run the tests with local SYCL RT instead
# of using the release RT included in this image.
ADD --chown=sycl:sycl toolchain.tar.zst /__w/llvm/llvm/toolchain
ADD --chown=sycl:sycl e2e_binaries.tar.zst /__w/llvm/llvm/build-e2e
ADD --chown=sycl:sycl e2e_sources.tar.zst /__w/llvm/llvm/llvm
ADD --chown=sycl:sycl sycl_cts_bin.tar.zst /__w/llvm/llvm/build-cts

# Since `/__w/llvm/llvm` above is overriden by GHA, need to provide the
# following for using in CI:
COPY e2e_binaries.tar.zst /sycl-prebuilt/
COPY e2e_sources.tar.zst /sycl-prebuilt/
COPY sycl_cts_bin.tar.zst /sycl-prebuilt/
COPY toolchain.tar.zst /sycl-prebuilt/

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh

USER sycl
