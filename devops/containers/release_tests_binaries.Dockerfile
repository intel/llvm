ARG base_tag=alldeps
ARG base_image=ghcr.io/intel/llvm/ubuntu2404_intel_drivers

FROM $base_image:$base_tag

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

# Since `/__w/llvm/llvm` above is overriden by GHA, need to provide the
# following for using in CI:
COPY e2e_binaries.tar.zst /sycl-prebuilt/
COPY e2e_sources.tar.zst /sycl-prebuilt/
COPY toolchain.tar.zst /sycl-prebuilt/

COPY scripts/drivers_entrypoint.sh /drivers_entrypoint.sh
COPY scripts/create-sycl-user.sh /user-setup.sh
RUN /user-setup.sh
