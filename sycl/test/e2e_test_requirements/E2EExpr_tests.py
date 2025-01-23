from E2EExpr import E2EExpr

import unittest


class TestE2EExpr(unittest.TestCase):
    def test_basic(self):
        # Non build-only expressions should work the same
        self.assertTrue(E2EExpr.evaluate("linux", {"linux", "rt_feature"}, False))
        self.assertTrue(E2EExpr.evaluate("rt_feature", {"linux", "rt_feature"}, False))
        self.assertFalse(E2EExpr.evaluate( "another_aspect && rt_feature", {"linux", "rt_feature"}, False))
        # build-only expressions with no unknowns should work the same
        self.assertTrue(E2EExpr.evaluate("linux", {"linux"}, True, final_unknown_value=False))
        self.assertFalse(E2EExpr.evaluate( "linux && windows", {"linux"}, True, final_unknown_value=True))
        self.assertTrue(E2EExpr.evaluate( "!(windows || zstd)", {"linux"}, True, final_unknown_value=False))
        # build-only expressions where unknown affects the resulting value
        self.assertTrue(E2EExpr.evaluate("rt_feature", {}, True, final_unknown_value=True))
        self.assertFalse(E2EExpr.evaluate("rt_feature", {}, True, final_unknown_value=False))
        self.assertTrue(E2EExpr.evaluate("rt_feature", {"rt_feature"}, True, final_unknown_value=True))
        self.assertFalse(E2EExpr.evaluate("rt_feature", {"rt_feature"}, True, final_unknown_value=False))
        self.assertFalse(E2EExpr.evaluate("!rt_feature", {}, True, final_unknown_value=False))
        self.assertFalse(E2EExpr.evaluate("!!rt_feature", {}, True, final_unknown_value=False))
        self.assertTrue(E2EExpr.evaluate("windows || rt_feature", {"linux"}, True, final_unknown_value=True))
        self.assertFalse(E2EExpr.evaluate("windows || rt_feature", {"linux"}, True, final_unknown_value=False))
        self.assertTrue(E2EExpr.evaluate("linux && rt_feature", {"linux"}, True, final_unknown_value=True))
        self.assertFalse(E2EExpr.evaluate("linux && rt_feature", {"linux"}, True, final_unknown_value=False))
        self.assertTrue(E2EExpr.evaluate( "linux && !(windows || rt_feature)", {"linux"}, True, final_unknown_value=True))
        self.assertFalse(E2EExpr.evaluate( "linux && !(windows || rt_feature)", {"linux"}, True, final_unknown_value=False))
        # build-only expressions where unknown does not affect the resulting value
        self.assertTrue(E2EExpr.evaluate("linux || rt_feature", {"linux"}, True, final_unknown_value=True))
        self.assertTrue(E2EExpr.evaluate("linux || rt_feature", {"linux"}, True, final_unknown_value=False))
        self.assertFalse(E2EExpr.evaluate("windows && rt_feature", {"linux"}, True, final_unknown_value=True))
        self.assertFalse(E2EExpr.evaluate("windows && rt_feature", {"linux"}, True, final_unknown_value=False))
        self.assertFalse(E2EExpr.evaluate( "linux && (windows && rt_feature)", {"linux"}, True, final_unknown_value=True))
        self.assertFalse(E2EExpr.evaluate( "linux && (windows && rt_feature)", {"linux"}, True, final_unknown_value=False))


if __name__ == "__main__":
    unittest.main()
