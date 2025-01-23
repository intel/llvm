from lit.BooleanExpression import BooleanExpression


class E2EExpr(BooleanExpression):
    build_specific_features = {
        "build-and-run-mode",
        "target-spir",
        "target-nvidia",
        "target-amd",
        "target-native_cpu",
        "linux",
        "system-linux",
        "windows",
        "system-windows",
        "enable-perf-tests",
        "opencl_icd",
        "cuda_dev_kit",
        "zstd",
        "vulkan",
        "true",
        "false",
    }

    def __init__(self, string, variables, build_only_mode, findal_unknown_value):
        BooleanExpression.__init__(self, string, variables)
        self.build_only_mode = build_only_mode
        self.unknown = False
        self.findal_unknown_value = findal_unknown_value

    @staticmethod
    def evaluate(string, variables, build_only_mode, final_unknown_value=True):
        """
        string: Expression to evaluate
        variables: variables that evaluate to true
        build_only_mode: if true enables unknown values
        findal_unknown_value: findal boolean result if evaluation results in `unknown`
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
        return self.findal_unknown_value if self.unknown else self.value


import unittest


class TestE2EExpr(unittest.TestCase):
    def test_basic(self):
        # Non build-only expressions should work the same
        self.assertTrue(E2EExpr.evaluate("linux", {"linux", "rt_feature"}, False))
        self.assertTrue(E2EExpr.evaluate("rt_feature", {"linux", "rt_feature"}, False))
        self.assertFalse(
            E2EExpr.evaluate(
                "another_aspect && rt_feature", {"linux", "rt_feature"}, False
            )
        )
        # build-only expressions with no unknowns should work the same
        self.assertTrue(
            E2EExpr.evaluate("linux", {"linux"}, True, final_unknown_value=False)
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && windows", {"linux"}, True, final_unknown_value=True
            )
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "!(windows || zstd)", {"linux"}, True, final_unknown_value=False
            )
        )
        # build-only expressions where unknown affects the resulting value
        self.assertTrue(
            E2EExpr.evaluate("rt_feature", {}, True, final_unknown_value=True)
        )
        self.assertFalse(
            E2EExpr.evaluate("rt_feature", {}, True, final_unknown_value=False)
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "rt_feature", {"rt_feature"}, True, final_unknown_value=True
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "rt_feature", {"rt_feature"}, True, final_unknown_value=False
            )
        )
        self.assertFalse(
            E2EExpr.evaluate("!rt_feature", {}, True, final_unknown_value=False)
        )
        self.assertFalse(
            E2EExpr.evaluate("!!rt_feature", {}, True, final_unknown_value=False)
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "windows || rt_feature", {"linux"}, True, final_unknown_value=True
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "windows || rt_feature", {"linux"}, True, final_unknown_value=False
            )
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "linux && rt_feature", {"linux"}, True, final_unknown_value=True
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && rt_feature", {"linux"}, True, final_unknown_value=False
            )
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "linux && !(windows || rt_feature)",
                {"linux"},
                True,
                final_unknown_value=True,
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && !(windows || rt_feature)",
                {"linux"},
                True,
                final_unknown_value=False,
            )
        )
        # build-only expressions where unknown does not affect the resulting value
        self.assertTrue(
            E2EExpr.evaluate(
                "linux || rt_feature", {"linux"}, True, final_unknown_value=True
            )
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "linux || rt_feature", {"linux"}, True, final_unknown_value=False
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "windows && rt_feature", {"linux"}, True, final_unknown_value=True
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "windows && rt_feature", {"linux"}, True, final_unknown_value=False
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && (windows && rt_feature)",
                {"linux"},
                True,
                final_unknown_value=True,
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && (windows && rt_feature)",
                {"linux"},
                True,
                final_unknown_value=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
