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

    def __init__(self, string, variables, build_only_mode, ignore_value):
        BooleanExpression.__init__(self, string, variables)
        self.build_only_mode = build_only_mode
        self.ignore = False
        self.ignore_value = ignore_value

    @staticmethod
    def evaluate(string, variables, build_only_mode, ignore_value=True):
        try:
            parser = E2EExpr(string, set(variables), build_only_mode, ignore_value)
            return parser.parseAll()
        except ValueError as e:
            raise ValueError(str(e) + ("\nin expression: %r" % string))

    def parseMATCH(self):
        token = self.token
        BooleanExpression.parseMATCH(self)
        if token not in self.build_specific_features and self.build_only_mode:
            self.ignore = True
        else:
            self.ignore = False

    def parseAND(self):
        self.parseNOT()
        while self.accept("&&"):
            left = self.value
            left_ignore = self.ignore
            self.parseNOT()
            right = self.value
            right_ignore = self.ignore
            self.value = left and right
            # Ignore if both are ignore or if one is true and the other is ignore
            self.ignore = (
                (left_ignore and right_ignore)
                or (left_ignore and right)
                or (left and right_ignore)
            )

    def parseOR(self):
        self.parseAND()
        while self.accept("||"):
            left = self.value
            left_ignore = self.ignore
            self.parseAND()
            right = self.value
            right_ignore = self.ignore
            self.value = left or right
            # Ignore if both are ignore or if one is false and the other is ignore
            self.ignore = (
                (left_ignore and right_ignore)
                or (left_ignore and not right)
                or (not left and right_ignore)
            )

    def parseAll(self):
        self.token = next(self.tokens)
        self.parseOR()
        self.expect(BooleanExpression.END)
        return self.ignore_value if self.ignore else self.value


import unittest


class TestE2EExpr(unittest.TestCase):
    def test_basic(self):
        # Non build-only expressions should work the same
        self.assertTrue(E2EExpr.evaluate("linux", {"linux", "rt_feature"}, False))
        self.assertTrue(
            E2EExpr.evaluate("rt_feature", {"linux", "rt_feature"}, False)
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "another_aspect && rt_feature", {"linux", "rt_feature"}, False
            )
        )
        # build-only expressions with no ignores should work the same
        self.assertTrue(
            E2EExpr.evaluate("linux", {"linux", "rt_feature"}, True, False)
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && windows", {"linux", "rt_feature"}, True, True
            )
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "!(windows || zstd)", {"linux", "rt_feature"}, True, False
            )
        )
        # build-only expressions where ignore affects the resulting value
        self.assertTrue(
            E2EExpr.evaluate("rt_feature", {"rt_feature"}, True, True)
        )
        self.assertFalse(
            E2EExpr.evaluate("rt_feature", {"rt_feature"}, True, False)
        )
        self.assertTrue(E2EExpr.evaluate("rt_feature", {}, True, True))
        self.assertTrue(
            E2EExpr.evaluate("!rt_feature", {"rt_feature"}, True, True)
        )
        self.assertTrue(
            E2EExpr.evaluate("!!rt_feature", {"rt_feature"}, True, True)
        )
        self.assertTrue(
            E2EExpr.evaluate("windows || rt_feature", {"linux"}, True, True)
        )
        self.assertFalse(
            E2EExpr.evaluate("windows || rt_feature", {"linux"}, True, False)
        )
        self.assertTrue(
            E2EExpr.evaluate("linux && rt_feature", {"linux"}, True, True)
        )
        self.assertFalse(
            E2EExpr.evaluate("linux && rt_feature", {"linux"}, True, False)
        )
        self.assertTrue(
            E2EExpr.evaluate(
                "linux && !(windows || rt_feature)", {"linux"}, True, True
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && !(windows || rt_feature)", {"linux"}, True, False
            )
        )
        # build-only expressions where ignore does not affect the resulting value
        self.assertTrue(
            E2EExpr.evaluate("linux || rt_feature", {"linux"}, True, True)
        )
        self.assertTrue(
            E2EExpr.evaluate("linux || rt_feature", {"linux"}, True, False)
        )
        self.assertFalse(
            E2EExpr.evaluate("windows && rt_feature", {"linux"}, True, True)
        )
        self.assertFalse(
            E2EExpr.evaluate("windows && rt_feature", {"linux"}, True, False)
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && (windows && rt_feature)", {"linux"}, True, True
            )
        )
        self.assertFalse(
            E2EExpr.evaluate(
                "linux && (windows && rt_feature)", {"linux"}, True, False
            )
        )


if __name__ == "__main__":
    unittest.main()
