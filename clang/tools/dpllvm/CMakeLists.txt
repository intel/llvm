set(LLVM_LINK_COMPONENTS
  Support
  )

# dpllvm is the entry-point shim for the DPC++ toolchain.  It is installed into
# bin/ next to the real tools; the public "dp"-prefixed names are symlinks
# (copies on Windows, which lacks first-class symlinks) to it.  At runtime
# dpllvm strips the "dp" prefix from argv[0] and execs the sibling real binary
# (dpclang -> clang, dpclang++ -> clang++, dpsycl-ls -> sycl-ls, ...).
#
# These names must not collide with a system LLVM/Clang on PATH, hence the
# prefix; the real binaries keep their original names because the clang driver
# re-invokes itself by name and expects to find "clang"/"clang++".
add_clang_tool(dpllvm
  dpllvm.cpp
  )

# The public entry-points.  Each is a symlink/copy to dpllvm; the stripped name
# (after removing "dp") must match a real binary installed next to dpllvm.
#   dpclang*        -> clang, clang++, clang-cl, clang-cpp  (this project)
#   dpsycl-ls       -> sycl-ls        (sycl/tools/sycl-ls)
#   dpsyclbin-dump  -> syclbin-dump   (sycl/tools/syclbin-dump)
#   dpsycl-prof     -> sycl-prof      (sycl/tools/sycl-prof, UNIX + XPTI only)
#   dpsycl-sanitize -> sycl-sanitize  (sycl/tools/sycl-sanitize, UNIX + XPTI)
#   dpsycl-trace    -> sycl-trace     (sycl/tools/sycl-trace, UNIX + XPTI)
#
# add_clang_symlink creates both the build-tree link (POST_BUILD of dpllvm, so
# it lands in the same build bin/ as every real tool) and the install rule
# (component "dpllvm").  The sycl-* entry-points are created here rather than in
# their own tool directories so that all symlink creation lives in one place and
# ships as a single component that deploy-sycl-toolchain pulls in; a symlink
# whose target tool is not built on a given platform simply errors at runtime,
# which is the same behavior as a missing tool.
set(DPCPP_ENTRY_POINTS
  dpclang
  dpclang++
  dpclang-cl
  dpclang-cpp
  dpsycl-ls
  dpsyclbin-dump
  dpsycl-prof
  dpsycl-sanitize
  dpsycl-trace
  )

foreach(link ${DPCPP_ENTRY_POINTS})
  add_clang_symlink(${link} dpllvm)
endforeach()
