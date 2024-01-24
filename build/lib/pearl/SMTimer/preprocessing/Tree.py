import numpy as np

pysmt_op = ["forall", "exists", "and", "or", "not", "implies", "iff", "symbol", "function", "real_constant", "bool_constant",
      "int_constant", "str_constant", "+", "-", "*", "<=", "<", "=", "ite", "toreal", "bv_constant", "bvnot", "bvand",
      "bvor", "bvxor", "concat", "extract", "bvult", "bvule", "bvneg", "bvadd", "bvsub", "bvmul", "bvudiv", "bvurem",
      "bvshl", "bvlshr", "bvrol", "bvror", "zero_extend", "sign_extend", "bvslt", "bvsle", "bvcomp", "bvsdiv", "bvsrem",
      "bvashr", "str.len", "str.++", "str.contains", "str.indexof", "str.replace", "str.substr", "str.prefixof",
      "str.suffixof", "str.to_int", "str.from_int", "str.at", "select", "store", "value", "/", "^",
      "algebraic_constant", "bv2nat"]

other_op = ["compressed_op", "unknown", "distinct", ">=", ">", "bvuge", "bvugt", "bvsge", "bvsgt", "str.in_re",
          "str.to_re", "to_fp",  "re.range", "re.union", "re.++", "re.+", "re.*", "re.allchar", "re.none"]

fp_op = ["fp", "fp.neg", "fp.isZero", "fp.isNormal", "fp.isSubnormal", "fp.isPositive", "fp.isInfinite", "fp.isNan",
         "fp.eq", "fp.roundToIntegral", "fp.rem", "fp.sub", "fp.sqrt", "fp.lt", "fp.leq", "fp.gt", "fp.geq", "fp.abs",
         "fp.add", "fp.div", "fp.min", "fp.max", "fp.mul", "fp.to_sbv", "fp.to_ubv"]

op = pysmt_op + other_op

none_op = ["extract", "zero_extend", "sign_extend", "to_fp", "repeat", "+oo", "-oo"]

tri_op = ["ite", "str.indexof", "str.replace", "str.substr", "store"]

bv_constant = "constant"
bool_constant = "constant"


class Tree:
    def __init__(self, val, left= None, mid= None, right= None):
        if isinstance(val, str):
            if val in op:
                pass
            elif val == "constant":
                self.name = "constant"
            elif val.startswith("var"):
                self.name = "var"
            else:
                self.name = "mid_val"
        elif isinstance(val, Tree):
            raise TypeError("The first argument should be string value of tree node, not tree instance")
        self.val = val
        for child in [left, mid, right]:
            if child and not isinstance(child,Tree):
                # print(child)
                raise ValueError("tree child is not a tree instance")
        self.left = left
        self.mid = mid
        self.right = right
        self.name = None

    def set_name(self, name):
        self.name = name

    def __str__(self):
        left_val = ""
        if self.left and self.left.name:
            left_val = self.left.name
        mid_val = ""
        if self.mid and self.mid.name:
            mid_val = self.mid.name
        right_val = ""
        if self.right and self.right.name:
            right_val = self.right.name
        name = ""
        if self.name:
            name = self.name
        if self.val == "concat":
            mid_val = "mid_val"
        return (' '.join([name,"(",self.val, left_val, mid_val, right_val, ")"]))

class varTree(Tree):
    def __init__(self, val, left= None, mid= None, right= None):
        super(varTree,self).__init__(val, left, mid, right)
        self.var = set()
        self.depth = 0
        self.compress_depth = 0
        self.node = 1
        self.feature = np.zeros(150)
        if val.startswith("var"):
            self.var.add(val)
            ind = min(int(val[3:]), 20)
            self.feature[111 + ind] += 1
        elif val in op:
            self.feature[op.index(val)] = 1

    def cal(self):
        for child in [self.left, self.mid, self.right]:
            if child:
                self.update(child)
        # if self.val == "concat":
        #     self.reduce_concat()
        #     self.compress_depth -= 1

    def update(self, child):
        self.depth = max(self.depth, child.depth + 1)
        self.compress_depth = max(self.compress_depth, child.compress_depth + 1)
        self.node = self.node + child.node
        self.var.update(child.var)
        self.feature += child.feature

    def reduce_concat(self):
        if not self.left:
            raise ValueError
        if not self.mid:
            raise ValueError
        var = set()
        var.update(self.left.var)
        var.update(self.mid.var)
        if self.left.var == var and self.left.depth >= self.mid.depth:
            self.replace_children(self.left)
        elif self.mid.var == var and self.left.depth <= self.mid.depth:
            self.replace_children(self.mid)
        elif self.left.depth > self.mid.depth:
            self.replace_children(self.left)
        else:
            self.replace_children(self.mid)

    def replace_children(self, tree):
        self.val = tree.val
        left, mid, right = tree.left, tree.mid, tree.right
        self.left, self.mid, self.right = left, mid, right

    def __str__(self):
        n = super(varTree, self).__str__()
        return " ".join([n, "depth:", str(self.depth), "compress_depth:" , str(self.compress_depth)])
