from lit.BooleanExpression import BooleanExpression


class E2EExpr(BooleanExpression):
    build_specific_features = {
        "run-mode",
        "build-mode",
        "preview-mode",
        "target-spir",
        "target-nvidia",
        "target-amd",
        "target-native_cpu",
        "any-target-is-spir",
        "any-target-is-nvidia",
        "any-target-is-amd",
        "any-target-is-native_cpu",
        "opencl-cpu-rt",
        "spirv-backend",
        "linux",
        "system-linux",
        "windows",
        "system-windows",
        "cl_options",
        "enable-perf-tests",
        "preview-breaking-changes-supported",
        "has_ndebug",
        "ocloc",
        "opencl-aot",
        "opencl_icd",
        "cm-compiler",
        "xptifw",
        "level_zero_dev_kit",
        "cuda_dev_kit",
        "hip_dev_kit",
        "zstd",
        "vulkan",
        "hip_options",
        "cuda_options",
        "host=None",
        "target=None",
        "shell",
        "non-root-user",
        "llvm-spirv",
        "llvm-link",
        "true",
        "false",
        "pdtracker",
    }

    def __init__(self, string, variables, build_only_mode, final_unknown_value):
        BooleanExpression.__init__(self, string, variables)
        self.build_only_mode = build_only_mode
        self.unknown = False
        self.final_unknown_value = final_unknown_value

    @staticmethod
    def evaluate(string, variables, build_only_mode, final_unknown_value=True):
        """
        string: Expression to evaluate
        variables: variables that evaluate to true
        build_only_mode: if true enables unknown values
        final_unknown_value: final boolean result if evaluation results in `unknown`
        """
        try:
            parser = E2EExpr(
                string, set(variables), build_only_mode, final_unknown_value
            )
            return parser.parseAll()
        except ValueError as e:
            raise ValueError(str(e) + ("\nin expression: %r" % string))

    def parseMATCH(self):
        token = self.token
        BooleanExpression.parseMATCH(self)
        if token not in E2EExpr.build_specific_features and self.build_only_mode:
            self.unknown = True
        else:
            self.unknown = False

    def parseAND(self):
        self.parseNOT()
        while self.accept("&&"):
            left = self.value
            left_unknown = self.unknown
            self.parseNOT()
            right = self.value
            right_unknown = self.unknown
            self.value = left and right
            # Unknown if both are unknown or if one is true and the other is unknown
            self.unknown = (
                (left_unknown and right_unknown)
                or (left_unknown and right)
                or (left and right_unknown)
            )

    def parseOR(self):
        self.parseAND()
        while self.accept("||"):
            left = self.value
            left_unknown = self.unknown
            self.parseAND()
            right = self.value
            right_unknown = self.unknown
            self.value = left or right
            # Unknown if both are unknown or if one is false and the other is unknown
            self.unknown = (
                (left_unknown and right_unknown)
                or (left_unknown and not right)
                or (not left and right_unknown)
            )

    def parseAll(self):
        self.token = next(self.tokens)
        self.parseOR()
        self.expect(BooleanExpression.END)
        return self.final_unknown_value if self.unknown else self.value

    @staticmethod
    def check_build_features(variables):
        rt_features = [x for x in variables if x not in E2EExpr.build_specific_features]
        if rt_features:
            raise ValueError(
                "Runtime features: "
                + str(rt_features)
                + " evaluated to True in build-only\n"
                + "If this is a new build specific feature append it to:"
                + "`build_specific_features` in `sycl/test-e2e/E2EExpr.py`"
            )


import unittest


class TestE2EExpr(unittest.TestCase):
    def test_basic(self):
        BuildOnly = True
        BuildAndRun = False
        RequiresDirective = True
        UnsupportedDirective = False
        RegularEval = lambda expr, features: E2EExpr.evaluate(
            expr, features, BuildAndRun
        )
        RequiresBuildEval = lambda expr, features: E2EExpr.evaluate(
            expr, features, BuildOnly, RequiresDirective
        )
        UnsupportedBuildEval = lambda expr, features: E2EExpr.evaluate(
            expr, features, BuildOnly, UnsupportedDirective
        )
        # Non build-only expressions should work the same
        self.assertTrue(RegularEval("linux", {"linux", "rt_feature"}))
        self.assertTrue(RegularEval("rt_feature", {"linux", "rt_feature"}))
        self.assertFalse(
            RegularEval("rt_feature1 && rt_feature2", {"linux", "rt_feature1"})
        )
        # build-only expressions with no unknowns should work the same
        self.assertTrue(UnsupportedBuildEval("linux", {"linux"}))
        self.assertFalse(RequiresBuildEval("linux && windows", {"linux"}))
        self.assertTrue(UnsupportedBuildEval("!(windows || zstd)", {"linux"}))
        # build-only expressions where unknown affects the resulting value
        self.assertTrue(RequiresBuildEval("rt_feature", {}))
        self.assertFalse(UnsupportedBuildEval("rt_feature", {}))
        self.assertFalse(UnsupportedBuildEval("!rt_feature", {}))
        self.assertTrue(RequiresBuildEval("windows || rt_feature", {"linux"}))
        self.assertFalse(UnsupportedBuildEval("windows || rt_feature", {"linux"}))
        self.assertTrue(RequiresBuildEval("linux && rt_feature", {"linux"}))
        self.assertFalse(UnsupportedBuildEval("linux && rt_feature", {"linux"}))
        self.assertTrue(RequiresBuildEval("linux && !(zstd || rt_feature)", {"linux"}))
        self.assertFalse(
            UnsupportedBuildEval("linux && !(zstd || rt_feature)", {"linux"})
        )
        # build-only expressions where unknown does not affect the resulting value
        self.assertTrue(RequiresBuildEval("linux || rt_feature", {"linux"}))
        self.assertTrue(UnsupportedBuildEval("linux || rt_feature", {"linux"}))
        self.assertFalse(RequiresBuildEval("windows && rt_feature", {"linux"}))
        self.assertFalse(UnsupportedBuildEval("windows && rt_feature", {"linux"}))
        self.assertFalse(
            RequiresBuildEval("linux && (vulkan && rt_feature)", {"linux"})
        )
        self.assertFalse(
            UnsupportedBuildEval("linux && (vulkan && rt_feature)", {"linux"})
        )
        # Check that no runtime features are present in build-only
        with self.assertRaises(ValueError):
            E2EExpr.check_build_features({"rt-feature"})
        with self.assertRaises(ValueError):
            E2EExpr.check_build_features({"build-only", "rt-feature"})
        E2EExpr.check_build_features({"build-mode"})


if __name__ == "__main__":
    unittest.main()
