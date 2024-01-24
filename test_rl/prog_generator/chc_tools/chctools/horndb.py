import sys
import z3
import io

def ground_quantifier(qexpr):
    body = qexpr.body()

    vars = list()
    for i in reversed(range(qexpr.num_vars())):
        vi_name = qexpr.var_name(i)
        vi_sort = qexpr.var_sort(i)
        vi = z3.Const(vi_name, vi_sort)
        vars.append(vi)

    body = z3.substitute_vars(body, *vars)
    return (body, vars)

def find_all_uninterp_consts(formula, res):
    if z3.is_quantifier(formula):
        formula = formula.body()

    worklist = []
    if z3.is_implies(formula):
        worklist.append(formula.arg(1))
        arg0 = formula.arg(0)
        if z3.is_and(arg0):
            worklist.extend(arg0.children())
        else:
            worklist.append(arg0)
    else:
        worklist.append(formula)

    for t in worklist:
        if z3.is_app(t) and t.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            res.append(t.decl())

class HornRule(object):
    def __init__(self, formula):
        self._formula = formula
        self._head = None
        self._body = []
        self._uninterp_sz = 0
        self._bound_constants = []

        self._update()

    def _update(self):
        if not self.has_formula():
            return

        rels = list()
        find_all_uninterp_consts(self._formula, rels)
        self._rels = frozenset(rels)
        body = self._formula
        if z3.is_quantifier(body):
            body, self._bound_constants = ground_quantifier(body)

        if z3.is_implies(body):
            self._head = body.arg(1)
            body = body.arg(0)
            if z3.is_and(body):
                body = body.children()
            else:
                body = [body]
        else:
            self._head = body
            body = []

        if len(body) > 0:
            self._body = body

        for i in range(len(body)):
            f = body[i]
            if z3.is_app(f) and f.decl() in self._rels:
                self._uninterp_sz += 1
            else:
                break

        assert(self._head is not None)

    def __str__(self):
        return str(self._formula)
    def __repr__(self):
        return repr(self._formula)

    def used_rels(self):
        return self._rels

    def is_query(self):
        return z3.is_false(self._head)

    def is_fact(self):
        return self._uninterp_sz == 0

    def is_linear(self):
        return self._uninterp_sz <= 1

    def to_ast(self):
        return self._formula

    def head(self):
        return self._head

    def body(self):
        return self._body

    def uninterp_size(self):
        return self._uninterp_sz

    def has_formula(self):
        return self._formula is not None

    def get_formula(self):
        return self._formula

    def mk_formula(self):
        f = self._body
        if len(f) == 0:
            f = z3.BoolVal(True)
        else:
            f = z3.And(f)
        f = z3.Implies(f, self._head)

        if len(self._bound_constants) > 0:
            f = z3.ForAll(self._bound_constants, f)
        self._formula = f
        return self._formula

    def mk_query(self):
        assert(self.is_query())
        f = self._body
        assert(len(f) > 0)
        f = z3.And(f)
        if len(self._bound_constants) > 0:
            f = z3.Exists(self._bound_constants, f)
        return f

class HornClauseDb(object):
    def __init__(self, name = 'horn'):
        self._name = name
        self._rules = []
        self._queries = []
        self._rels = frozenset()
        self._sealed = True
        self._fp = False

    def add_rule(self, horn_rule):
        self._sealed = False
        if horn_rule.is_query():
            self._queries.append(horn_rule)
        else:
            self._rules.append(horn_rule)

    def get_rels(self):
        self.seal()
        return self._rels

    def get_rules(self):
        return self._rules
    def get_queries(self):
        return self._queries

    def seal(self):
        if self._sealed:
            return

        rels = list()
        for r in self._rules:
            rels.extend(r.used_rels())
        for q in self._queries:
            rels.extend(r.used_rels())
        self._rels = frozenset(rels)
        self._sealed = True

    def __str__(self):
        out = io.StringIO()
        for r in self._rules:
            out.write(str(r))
            out.write('\n')
        out.write('\n')
        for q in self._queries:
            out.write(str(q))
        return out.getvalue()

    def load_from_fp(self, fp, queries):
        self._fp = fp
        if len(queries) > 0:
            for r in fp.get_rules():
                rule = HornRule(r)
                self.add_rule(rule)
            for q in queries:
                rule = HornRule(z3.Implies(q, z3.BoolVal(False)))
                self.add_rule(rule)
        else:
            # fixedpoit object is not properly loaded, ignore it
            self._fp = None
            for a in fp.get_assertions():
                rule = HornRule(a)
                self.add_rule(rule)
        self.seal()

    def has_fixedpoint(self):
        return self._fp is not None
    def get_fixedpoint(self):
        return self._fp

    def mk_fixedpoint(self, fp = None):
        if fp is None:
            self._fp = z3.Fixedpoint()
            fp = self._fp

        for rel in self._rels:
            fp.register_relation(rel)
        for r in self._rules:
            if r.has_formula():
                fp.add_rule(r.get_formula())
            else:
                fp.add_rule(r.mk_formula())

        return fp

class FolModel(object):
    def __init__(self):
        self._fn_interps = dict()

    def add_fn(self, name, lmbd):
        self._fn_interps[name] = lmbd

    def has_interp(self, name):
        return name in self._fn_interps.keys()

    def __setitem__(self, key, val):
        self.add_fn(key, val)

    def get_fn(self, name):
        # print("Fn", self._fn_interps)
        return self._fn_interps[name]

    def eval(self, term):
        fn = self.get_fn(term.decl().name())
        # lambdas seem to be broken at the moment
        # this is a work around
        body = fn.body()
        body = z3.substitute_vars(body, *reversed(term.children()))
        return body

    def get_fn_collection(self):
        return self._fn_interps

    def __str__(self):
        return str(self._fn_interps)
    def __repr__(self):
        return repr(self._fn_interps)

def load_horn_db_from_file(fname):
    fp = z3.Fixedpoint()
    queries = fp.parse_file(fname)
    db = HornClauseDb(fname)
    db.load_from_fp(fp, queries)
    return db

def main():
    db = load_horn_db_from_file(sys.argv[1])
    print(db)
    print(db.get_rels())
    return 0
if __name__ == '__main__':
    sys.exit(main())
