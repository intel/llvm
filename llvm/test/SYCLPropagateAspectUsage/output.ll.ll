; ModuleID = 'output.ll'
source_filename = "<stdin>"

%MyStruct = type { i32 }

define dso_local spir_kernel void @kernel1() !intel_used_aspects !4 {
  call spir_func void @func1()
  ret void
}

define weak dso_local spir_func void @func1() !intel_used_aspects !4 {
  %struct = alloca %MyStruct, align 8
  ret void
}

!intel_types_that_use_aspects = !{!0, !1, !2, !3}

!0 = !{!"class.cl::sycl::detail::half_impl::half", i32 8}
!1 = !{!"class.cl::sycl::amx_type", i32 9}
!2 = !{!"class.cl::sycl::other_type", i32 8, i32 9}
!3 = !{!"MyStruct", i32 8}
!4 = !{i32 8}
