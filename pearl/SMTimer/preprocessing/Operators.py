pysmt_op = ["forall", "exists", "and", "or", "not", "=>", "iff", "symbol", "function", "real_constant", "bool_constant",
      "int_constant", "str_constant", "+", "-", "*", "<=", "<", "=", "ite", "toreal", "bv_constant", "bvnot", "bvand",
      "bvor", "bvxor", "concat", "extract", "bvult", "bvule", "bvneg", "bvadd", "bvsub", "bvmul", "bvudiv", "bvurem",
      "bvshl", "bvlshr", "bvrol", "bvror", "zero_extend", "sign_extend", "bvslt", "bvsle", "bvcomp", "bvsdiv", "bvsrem",
      "bvashr", "str.len", "str.++", "str.contains", "str.indexof", "str.replace", "str.substr", "str.prefixof",
      "str.suffixof", "str.to_int", "str.from_int", "str.at", "select", "store", "value", "/", "^",
      "algebraic_constant", "bv2nat"]
other_op = ["compressed_op", "unknown", "distinct", ">=", ">", "bvuge", "bvugt", "bvsge", "bvsgt", "str.in_re",
          "str.to_re", "to_fp",  "re.range", "re.union", "re.++", "re.+", "re.*", "re.allchar", "re.none", "xor",
          "mod"]
fp_op = ["fp", "fp.neg", "fp.isZero", "fp.isNormal", "fp.isSubnormal", "fp.isPositive", "fp.isInfinite", "fp.isNan",
         "fp.eq", "fp.roundToIntegral", "fp.rem", "fp.sub", "fp.sqrt", "fp.lt", "fp.leq", "fp.gt", "fp.geq", "fp.abs",
         "fp.add", "fp.div", "fp.min", "fp.max", "fp.mul", "fp.to_sbv", "fp.to_ubv"]
op = pysmt_op + other_op
none_op = ["extract", "zero_extend", "sign_extend", "to_fp", "repeat", "+oo", "-oo"]
tri_op = ["ite", "str.indexof", "str.replace", "str.substr", "store"]
bv_constant = "constant"
bool_constant = "constant"
reserved_word = ["declare-fun", "define-fun", "declare-sort", "define-sort", "declare-datatype", "declare-const"
                 "assert", "check-sat", "set-info", "set-logic", "set-option"]