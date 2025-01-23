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
            parser = E2EExpr(string, set(variables), build_only_mode, final_unknown_value)
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
