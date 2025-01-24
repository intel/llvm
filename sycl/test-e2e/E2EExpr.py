from lit.BooleanExpression import BooleanExpression


class E2EExpr(BooleanExpression):
    build_specific_features = {
        "build-and-run-mode",
        "target-spir",
        "target-nvidia",
        "target-amd",
        "target-native_cpu",
        "any-target-is-spir",
        "any-target-is-nvidia",
        "any-target-is-amd",
        "any-target-is-native_cpu",
        "linux",
        "system-linux",
        "windows",
        "system-windows",
        "enable-perf-tests",
        "preview-breaking-changes-supported",
        "has_ndebug",
        "ocloc",
        "opencl-aot",
        "opencl_icd",
        "cm-compiler",
        "xptifw"
        "level_zero_dev_kit",
        "cuda_dev_kit",
        "zstd",
        "vulkan",
        "true",
        "false",
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
        if token not in self.build_specific_features and self.build_only_mode:
            self.unknown = True
        else:
            self.unknown = False
        if self.value and self.unknown:
            raise ValueError("Runtime feature \"" + token +"\" evaluated to True in build-only")

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


import unittest


class TestE2EExpr(unittest.TestCase):
    def test_basic(self):
        BuildOnly = True
        BuildAndRun = False
        RequiresDirective = True
        UnsupportedDirective = False
        RegularEval= lambda expr, features: E2EExpr.evaluate(expr, features, BuildAndRun)
        RequiresBuildEval = lambda expr, features: E2EExpr.evaluate(expr, features, BuildOnly, RequiresDirective)
        UnsupportedBuildEval = lambda expr, features: E2EExpr.evaluate(expr, features, BuildOnly, UnsupportedDirective)
        # Non build-only expressions should work the same
        self.assertTrue(RegularEval("linux", {"linux", "rt_feature"}))
        self.assertTrue(RegularEval("rt_feature", {"linux", "rt_feature"}))
        self.assertFalse(RegularEval("rt_feature1 && rt_feature2", {"linux", "rt_feature1"}))
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
        self.assertFalse(UnsupportedBuildEval("linux && !(zstd || rt_feature)", {"linux"}))
        # build-only expressions where unknown does not affect the resulting value
        self.assertTrue(RequiresBuildEval("linux || rt_feature", {"linux"}))
        self.assertTrue(UnsupportedBuildEval("linux || rt_feature", {"linux"}))
        self.assertFalse(RequiresBuildEval("windows && rt_feature", {"linux"}))
        self.assertFalse(UnsupportedBuildEval("windows && rt_feature", {"linux"}))
        self.assertFalse(RequiresBuildEval("linux && (vulkan && rt_feature)", {"linux"}))
        self.assertFalse(UnsupportedBuildEval("linux && (vulkan && rt_feature)", {"linux"}))
        # runtime feature is present in build-only
        with self.assertRaises(ValueError):
            RequiresBuildEval("rt_feature", {"rt_feature"})
        with self.assertRaises(ValueError):
            UnsupportedBuildEval("rt_feature", {"rt_feature"})


if __name__ == "__main__":
    unittest.main()
