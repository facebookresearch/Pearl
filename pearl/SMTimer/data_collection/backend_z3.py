
import os
import z3
import ctypes
import logging
import numbers
import operator
import threading
import weakref
from functools import reduce
from decimal import Decimal

from cachetools import LRUCache

from ..errors import ClaripyZ3Error
from .output_query_data_struct import query_data
import time

l = logging.getLogger("claripy.backends.backend_z3")

#pylint:disable=unidiomatic-typecheck

try:
    import __pypy__
    _is_pypy = True
except ImportError:
    _is_pypy = False

def z3_expr_to_smt2(f, status="unknown", name="benchmark", logic=""):
      # from https://stackoverflow.com/a/14629021/9719920
      v = (z3.Ast * 0)()
      return z3.Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())

def claripy_solver_to_smt2(s):
    return s._get_solver().to_smt2()

def _add_memory_pressure(p):
    """
    PyPy's garbage collector is not aware of memory uses happening inside native code. When performing memory-intensive
    tasks in native code, the memory pressure that PyPy observes can greatly deviate from the actual memory pressure.
    We must manually add sufficient memory pressure to account for the "missing" portion.

    This is not a problem for CPython since its GC is based on reference counting.
    """

    global _is_pypy
    if _is_pypy:
        __pypy__.add_memory_pressure(p)

#
# Some global variables
#

# track the count of solves
solve_count = 0

supports_fp = hasattr(z3, 'fpEQ')

# you can toggle this flag if you want. I don't think it matters
#z3.set_param('rewriter.hi_fp_unspecified', 'true')

#
# Utility functions
#

def condom(f):
    def z3_condom(*args, **kwargs):
        """
        The Z3 condom intercepts Z3Exceptions and throws a ClaripyZ3Error instead.
        """
        try:
            return f(*args, **kwargs)
        except z3.Z3Exception as ze:
            raise ClaripyZ3Error() from ze
    return z3_condom

def _raw_caller(f):
    @staticmethod
    @condom
    def raw_caller(*args, **kwargs):
        return f(*args, **kwargs)
    return raw_caller

def _z3_decl_name_str(ctx, decl):
    decl_name = z3.Z3_get_decl_name(ctx, decl)
    symbol_name = z3.Z3_get_symbol_string_bytes(ctx, decl_name)
    return symbol_name


class SmartLRUCache(LRUCache):
    def __init__(self, maxsize, getsizeof=None, evict=None):
        LRUCache.__init__(self, maxsize, getsizeof=getsizeof)
        self._evict = evict

    def popitem(self):
        key, val = LRUCache.popitem(self)
        if self._evict:
            self._evict(key, val)
        return key, val


#
# And the (ugh) magic
#

from . import Backend
class BackendZ3(Backend):
    _split_on = { 'And', 'Or' }
    query_record = query_data()

    def __init__(self, reuse_z3_solver=None, ast_cache_size=10000):
        Backend.__init__(self, solver_required=True)
        self._enable_simplification_cache = False
        self._hash_to_constraint = weakref.WeakValueDictionary()

        # Per-thread Z3 solver
        # This setting is treated as a global setting and is not supposed to be changed during runtime, unless you know
        # what you are doing.
        if reuse_z3_solver is None:
            reuse_z3_solver = True if os.environ.get('REUSE_Z3_SOLVER', "False").lower() in {"1", "true", "yes", "y"} \
                else False
        self.reuse_z3_solver = reuse_z3_solver

        self._ast_cache_size = ast_cache_size

        # and the operations
        all_ops = backend_fp_operations | backend_operations if supports_fp else backend_operations
        for o in all_ops - {'BVV', 'BoolV', 'FPV', 'FPS', 'BitVec', 'StringV'}:
            self._op_raw[o] = getattr(self, '_op_raw_' + o)
        self._op_raw['Xor'] = self._op_raw_Xor

        self._op_raw['__ge__'] = self._op_raw_UGE
        self._op_raw['__gt__'] = self._op_raw_UGT
        self._op_raw['__le__'] = self._op_raw_ULE
        self._op_raw['__lt__'] = self._op_raw_ULT

        self._op_raw['Reverse'] = self._op_raw_Reverse
        self._op_raw['Identical'] = self._identical
        self._op_raw['fpToSBV'] = self._op_raw_fpToSBV
        self._op_raw['fpToUBV'] = self._op_raw_fpToUBV

        self._op_expr['BVS'] = self.BVS
        self._op_expr['BVV'] = self.BVV
        self._op_expr['FPV'] = self.FPV
        self._op_expr['FPS'] = self.FPS
        self._op_expr['BoolV'] = self.BoolV
        self._op_expr['BoolS'] = self.BoolS
        self._op_expr['StringV'] = self.StringV
        self._op_expr['StringS'] = self.StringS

        self._op_raw['__floordiv__'] = self._op_div
        self._op_raw['__mod__'] = self._op_mod

        # reduceable
        self._op_raw['__add__'] = self._op_add
        self._op_raw['__sub__'] = self._op_sub
        self._op_raw['__mul__'] = self._op_mul
        self._op_raw['__or__'] = self._op_or
        self._op_raw['__xor__'] = self._op_xor
        self._op_raw['__and__'] = self._op_and

    # XXX this is a HUGE HACK that should be removed whenever uninitialized gets moved to the
    # "proposed annotation backend" or wherever will prevent it from being part of the object
    # identity. also whenever the VSA attributes get the fuck out of BVS as well
    @property
    def extra_bvs_data(self):
        try:
            return self._tls.extra_bvs_data
        except AttributeError:
            # a pointer to get values out of Z3
            self._tls.extra_bvs_data = {}
            return self._tls.extra_bvs_data


    @property
    def _c_uint64_p(self):
        try:
            return self._tls.c_uint64_p
        except AttributeError:
            # a pointer to get values out of Z3
            self._tls.c_uint64_p = ctypes.pointer(ctypes.c_uint64())

            return self._tls.c_uint64_p

    @property
    def _context(self):
        try:
            return self._tls.context
        except AttributeError:
            self._tls.context = z3.Context() if threading.current_thread().name != 'MainThread' else z3.main_ctx()
            return self._tls.context

    @property
    def _boolref_tactics(self):
        try:
            return self._tls.boolref_tactics
        except AttributeError:
            tactics = z3.Then(
                z3.Tactic("simplify", ctx=self._context),
                z3.Tactic("propagate-ineqs", ctx=self._context),
                z3.Tactic("propagate-values", ctx=self._context),
                z3.Tactic("unit-subsume-simplify", ctx=self._context),
                z3.Tactic("aig", ctx=self._context),
                ctx=self._context
            )
            self._tls.boolref_tactics = tactics
            return self._tls.boolref_tactics

    @property
    def _ast_cache(self):
        try:
            return self._tls.ast_cache
        except AttributeError:
            self._tls.ast_cache = SmartLRUCache(self._ast_cache_size, evict=self._pop_from_ast_cache)
            return self._tls.ast_cache

    @property
    def _var_cache(self):
        try:
            return self._tls.var_cache
        except AttributeError:
            self._tls.var_cache = weakref.WeakValueDictionary()
            return self._tls.var_cache

    @property
    def _sym_cache(self):
        try:
            return self._tls.sym_cache
        except AttributeError:
            self._tls.sym_cache = weakref.WeakValueDictionary()
            return self._tls.sym_cache

    @property
    def _simplification_cache_key(self):
        try:
            return self._tls.simplification_cache_key
        except AttributeError:
            self._tls.simplification_cache_key = weakref.WeakValueDictionary()
            return self._tls.simplification_cache_key

    @property
    def _simplification_cache_val(self):
        try:
            return self._tls.simplification_cache_val
        except AttributeError:
            self._tls.simplification_cache_val = weakref.WeakValueDictionary()
            return self._tls.simplification_cache_val

    def downsize(self):
        Backend.downsize(self)

        self._ast_cache.clear()
        self._var_cache.clear()
        self._sym_cache.clear()
        self._simplification_cache_key.clear()
        self._simplification_cache_val.clear()

    @condom
    def _size(self, a):
        if not isinstance(a, z3.BitVecRef) and not isinstance(a, z3.BitVecNumRef):
            l.debug("Unable to determine length of value of type %s", a.__class__)
            raise BackendError("Unable to determine length of value of type %s" % a.__class__)
        return a.size()

    def _name(self, o): #pylint:disable=unused-argument
        l.warning("BackendZ3.name() called. This is weird.")
        raise BackendError("name is not implemented yet")

    def _pop_from_ast_cache(self, _, tpl):
        _, raw_ast = tpl
        z3.Z3_dec_ref(self._context.ctx, raw_ast)

    #
    # Core creation methods
    #

    @condom
    def BVS(self, ast):
        name = ast._encoded_name
        if hasattr(ast, 'annotations'):
            self.extra_bvs_data[name] = (ast.args, ast.annotations)
        else:
            self.extra_bvs_data[name] = (ast.args, None)
        size = ast.size()
        # TODO: Here we can use low level APIs because the check performed by the high level API always results in
        #       the else branch of the check. This evidence although comes from the execution of the angr and claripy
        #       test_rl suite so I'm not sure if this assumption would hold on 100% of the cases
        bv = z3.BitVecSortRef(z3.Z3_mk_bv_sort(self._context.ref(), size), self._context)
        expr = z3.BitVecRef(z3.Z3_mk_const(self._context.ref(), z3.to_symbol(name, self._context), bv.ast), self._context)
        #if mn is not None:
        #    expr = z3.If(z3.ULT(expr, mn), mn, expr, ctx=self._context)
        #if mx is not None:
        #    expr = z3.If(z3.UGT(expr, mx), mx, expr, ctx=self._context)
        #if stride is not None:
        #    expr = (expr // stride) * stride
        return expr

    @condom
    def BVV(self, ast):
        if ast.args[0] is None:
            raise BackendError("Z3 can't handle empty BVVs")

        size = ast.size()
        # TODO: Here there is no need to use low level API since teh high level API just perform some conversions which
        #       are mandatory to fix the types of the arguments requested by the low level API
        return z3.BitVecVal(ast.args[0], size, ctx=self._context)

    @condom
    def FPS(self, ast):
        sort_z3 = self._convert(ast.args[1])
        # TODO: Here is safe to use low level API since the only condition the hig level API check is if the context is
        #       None and this never happens here
        return z3.FPRef(
            z3.Z3_mk_const(self._context.ref(), z3.to_symbol(ast._encoded_name, self._context), sort_z3.ast),
            self._context)

    @condom
    def FPV(self, ast):
        val = str(ast.args[0])
        sort = self._convert(ast.args[1])
        if val in ("+oo", "+inf", "+Inf", 'inf'):
            return z3.FPNumRef(z3.Z3_mk_fpa_inf(sort.ctx_ref(), sort.ast, False), sort.ctx)
        elif val in ("-oo", "-inf", "-Inf"):
            return z3.FPNumRef(z3.Z3_mk_fpa_inf(sort.ctx_ref(), sort.ast, True), sort.ctx)
        elif val in ("0.0", "+0.0"):
            return z3.FPNumRef(z3.Z3_mk_fpa_zero(sort.ctx_ref(), sort.ast, False), sort.ctx)
        elif val == '-0.0':
            return z3.FPNumRef(z3.Z3_mk_fpa_zero(sort.ctx_ref(), sort.ast, True), sort.ctx)
        elif val in ("NaN", "nan"):
            return z3.FPNumRef(z3.Z3_mk_fpa_nan(sort.ctx_ref(), sort.ast), sort.ctx)
        else:
            better_val = str(Decimal(ast.args[0]))
            return z3.FPNumRef(z3.Z3_mk_numeral(self._context.ref(), better_val, sort.ast), self._context)

    @condom
    def BoolS(self, ast):
        return z3.BoolRef(
            z3.Z3_mk_const(
                self._context.ref(), z3.to_symbol(ast._encoded_name, self._context), z3.BoolSort(self._context).ast
            ),
            self._context)

    @condom
    def BoolV(self, ast): #pylint:disable=unused-argument
        # TODO: Here the checks performed by the high level API are mandatory before calling the low level API
        #       So we can keep the high level API call here
        return z3.BoolVal(ast.args[0], ctx=self._context)

    @condom
    def StringV(self, ast):
        return z3.StringVal(ast.args[0], ctx=self._context)

    @condom
    def StringS(self, ast):
        return z3.String(ast.args[0], ctx=self._context)
    #
    # Conversions
    #

    @condom
    def _convert(self, obj):  # pylint:disable=arguments-differ
        if isinstance(obj, FSort):
            return z3.FPSortRef(z3.Z3_mk_fpa_sort(self._context.ref(), obj.exp, obj.mantissa), self._context)
        elif isinstance(obj, RM):
            if obj == RM.RM_NearestTiesEven:
                return z3.FPRMRef(z3.Z3_mk_fpa_round_nearest_ties_to_even(self._context.ref()), self._context)
            elif obj == RM.RM_NearestTiesAwayFromZero:
                return z3.FPRMRef(z3.Z3_mk_fpa_round_nearest_ties_to_away(self._context.ref()), self._context)
            elif obj == RM.RM_TowardsPositiveInf:
                return z3.FPRMRef(z3.Z3_mk_fpa_round_toward_positive(self._context.ref()), self._context)
            elif obj == RM.RM_TowardsNegativeInf:
                return z3.FPRMRef(z3.Z3_mk_fpa_round_toward_negative(self._context.ref()), self._context)
            elif obj == RM.RM_TowardsZero:
                return z3.FPRMRef(z3.Z3_mk_fpa_round_toward_zero(self._context.ref()), self._context)
            else:
                raise BackendError("unrecognized rounding mode")
        elif obj is True:
            return z3.BoolRef(z3.Z3_mk_true(self._context.ref()), self._context)
        elif obj is False:
            return z3.BoolRef(z3.Z3_mk_false(self._context.ref()), self._context)
        elif isinstance(obj, (numbers.Number, str)):
            return obj
        elif hasattr(obj, '__module__') and obj.__module__ in ('z3', 'z3.z3'):
            return obj
        else:
            l.debug("BackendZ3 encountered unexpected type %s", type(obj))
            raise BackendError("unexpected type %s encountered in BackendZ3" % type(obj))

    def call(self, *args, **kwargs):  # pylint;disable=arguments-differ
        return Backend.call(self, *args, **kwargs)

    @condom
    def _abstract(self, e):
        #return self._abstract(z, split_on=split_on)[0]
        return self._abstract_internal(e.ctx.ctx, e.ast)

    @staticmethod
    def _z3_ast_hash(ast):
        """
        This is a better hashing function for z3 Ast objects. Z3_get_ast_hash() creates too many hash collisions.

        :param ast: A z3 Ast object.
        :return:    An integer - the hash.
        """

        return ast.value

    def _abstract_internal(self, ctx, ast, split_on=None):
        h = self._z3_ast_hash(ast)
        try:
            cached_ast, _ = self._ast_cache[h]
            return cached_ast
        except KeyError:
            pass

        decl = z3.Z3_get_app_decl(ctx, ast)
        decl_num = z3.Z3_get_decl_kind(ctx, decl)
        z3_sort = z3.Z3_get_sort(ctx, ast)

        if decl_num not in z3_op_nums:
            raise ClaripyError("unknown decl kind %d" % decl_num)
        if z3_op_nums[decl_num] not in op_map:
            raise ClaripyError("unknown decl op %s" % z3_op_nums[decl_num])
        op_name = op_map[z3_op_nums[decl_num]]

        num_args = z3.Z3_get_app_num_args(ctx, ast)
        split_on = self._split_on if split_on is None else split_on
        new_split_on = split_on if op_name in split_on else set()
        children = [ self._abstract_internal(ctx, z3.Z3_get_app_arg(ctx, ast, i), new_split_on) for i in range(num_args) ]

        append_children = True

        if op_name == 'True':
            return BoolV(True)
        elif op_name == 'False':
            return BoolV(False)
        elif op_name.startswith('RM_'):
            return RM(op_name)
        elif op_name == 'BitVecVal':
            bv_size = z3.Z3_get_bv_sort_size(ctx, z3_sort)
            if z3.Z3_get_numeral_uint64(ctx, ast, self._c_uint64_p):
                return BVV(self._c_uint64_p.contents.value, bv_size)
            else:
                bv_num = int(z3.Z3_get_numeral_string(ctx, ast))
                return BVV(bv_num, bv_size)
        elif op_name in ('FPVal', 'MinusZero', 'MinusInf', 'PlusZero', 'PlusInf', 'NaN'):
            ebits = z3.Z3_fpa_get_ebits(ctx, z3_sort)
            sbits = z3.Z3_fpa_get_sbits(ctx, z3_sort)
            sort = FSort.from_params(ebits, sbits)
            val = self._abstract_fp_val(ctx, ast, op_name)
            return FPV(val, sort)

        elif op_name == 'UNINTERPRETED' and num_args == 0: # symbolic value
            symbol_name = _z3_decl_name_str(ctx, decl)
            symbol_str = symbol_name.decode()
            symbol_ty = z3.Z3_get_sort_kind(ctx, z3_sort)

            if symbol_ty == z3.Z3_BV_SORT:
                bv_size = z3.Z3_get_bv_sort_size(ctx, z3_sort)
                (ast_args, annots) = self.extra_bvs_data.get(symbol_name, (None, None))
                if ast_args is None:
                    ast_args = (symbol_str, None, None, None, False, False, None)

                return BV('BVS',
                        ast_args,
                        length=bv_size,
                        variables={ symbol_str },
                        symbolic=True,
                        encoded_name=symbol_name,
                        annotations=annots)
            elif symbol_ty == z3.Z3_BOOL_SORT:
                return Bool('BoolS',
                        (symbol_str,),
                        variables={ symbol_str },
                        symbolic=True)
            elif symbol_ty == z3.Z3_FLOATING_POINT_SORT:
                ebits = z3.Z3_fpa_get_ebits(ctx, z3_sort)
                sbits = z3.Z3_fpa_get_sbits(ctx, z3_sort)
                sort = FSort.from_params(ebits, sbits)
                return FP('FPS',
                        (symbol_str, sort),
                        variables={ symbol_str },
                        symbolic=True,
                        length=sort.length)
            else:
                raise BackendError("Unknown z3 term type %d...?" % symbol_ty)

        elif op_name == 'UNINTERPRETED':
            mystery_name = z3.Z3_get_symbol_string(ctx, z3.Z3_get_decl_name(ctx, decl))
            l.error("Mystery operation %s in BackendZ3._abstract_internal. Please report this.", mystery_name)
        elif op_name == 'Extract':
            hi = z3.Z3_get_decl_int_parameter(ctx, decl, 0)
            lo = z3.Z3_get_decl_int_parameter(ctx, decl, 1)
            args = [ hi, lo ]
        elif op_name in ('SignExt', 'ZeroExt'):
            num = z3.Z3_get_decl_int_parameter(ctx, decl, 0)
            args = [ num ]
        elif op_name in ('fpToFP', 'fpToFPSigned'):
            exp = z3.Z3_fpa_get_ebits(ctx, z3_sort)
            mantissa = z3.Z3_fpa_get_sbits(ctx, z3_sort)
            sort = FSort.from_params(exp, mantissa)
            args = children + [sort]
            append_children = False
        elif op_name in ('fpToSBV', 'fpToUBV'):
            # uuuuuugggggghhhhhh
            bv_size = z3.Z3_get_bv_sort_size(ctx, z3_sort)
            args = children + [bv_size]
            append_children = False
        else:
            args = [ ]

        if append_children:
            args.extend(children)

        # hmm.... honestly not sure what to do here
        result_ty = op_type_map[z3_op_nums[decl_num]]
        ty = type(args[-1])

        if type(result_ty) is str:
            err = "Unknown Z3 error in abstraction (result_ty == '%s'). Update your version of Z3, and, if the problem persists, open a claripy issue." % result_ty
            l.error(err)
            raise BackendError(err)

        if op_name == 'If':
            # If is polymorphic and thus must be handled specially
            ty = type(args[1])

            a = ty('If', tuple(args), length=args[1].length)
        elif hasattr(ty, op_name) or hasattr(_all_operations, op_name):
            op = getattr(ty if hasattr(ty, op_name) else _all_operations, op_name)
            if op.calc_length is not None:
                length = op.calc_length(*args)
                a = result_ty(op_name, tuple(args), length=length)
            else:
                a = result_ty(op_name, tuple(args))
        else:
            a = result_ty(op_name, tuple(args))

        self._ast_cache[h] = (a, ast)
        z3.Z3_inc_ref(ctx, ast)
        return a

    def _abstract_to_primitive(self, ctx, ast):
        decl = z3.Z3_get_app_decl(ctx, ast)
        decl_num = z3.Z3_get_decl_kind(ctx, decl)

        if decl_num not in z3_op_nums:
            raise ClaripyError("unknown decl kind %d" % decl_num)
        if z3_op_nums[decl_num] not in op_map:
            raise ClaripyError("unknown decl op %s" % z3_op_nums[decl_num])
        op_name = op_map[z3_op_nums[decl_num]]

        if op_name == 'BitVecVal':
            return self._abstract_bv_val(ctx, ast)
        elif op_name == 'True':
            return True
        elif op_name == 'False':
            return False
        elif op_name in ('FPVal', 'MinusZero', 'MinusInf', 'PlusZero', 'PlusInf', 'NaN'):
            return self._abstract_fp_val(ctx, ast, op_name)
        elif op_name == 'Concat':
            # Quirk in how z3 might handle NaN encodings - it will not give us a fully evaluated model
            # https://github.com/Z3Prover/z3/issues/518
            # this case will be triggered if the z3 rewriter.hi_fp_unspecified is set to true
            nargs = z3.Z3_get_app_num_args(ctx, ast)
            res = 0
            for i in range(nargs):
                arg_ast = z3.Z3_get_app_arg(ctx, ast, i)
                arg_decl = z3.Z3_get_app_decl(ctx, arg_ast)
                arg_decl_num = z3.Z3_get_decl_kind(ctx, arg_decl)
                arg_size = z3.Z3_get_bv_sort_size(ctx, z3.Z3_get_sort(ctx, arg_ast))

                neg = False
                if arg_decl_num == z3.Z3_OP_BNEG:
                    arg_ast = z3.Z3_get_app_arg(ctx, arg_ast, 0)
                    arg_decl = z3.Z3_get_app_decl(ctx, arg_ast)
                    arg_decl_num = z3.Z3_get_decl_kind(ctx, arg_decl)
                    neg = True
                if arg_decl_num != z3.Z3_OP_BNUM:
                    raise BackendError("Weird z3 model")

                arg_int = self._abstract_bv_val(ctx, arg_ast)
                if neg:
                    arg_int = (1<<arg_size)-arg_int
                res <<= arg_size
                res |= arg_int
            return res
        elif op_name == 'fpToIEEEBV':
            # Another quirk in the way z3 might handle nan encodings. see above
            # this case will be triggered if the z3 rewriter.hi_fp_unspecified is set to false
            arg_ast = z3.Z3_get_app_arg(ctx, ast, 0)
            return self._abstract_fp_encoded_val(ctx, arg_ast)
        else:
            raise BackendError("Unable to abstract Z3 object to primitive")

    def _abstract_bv_val(self, ctx, ast):
        if z3.Z3_get_numeral_uint64(ctx, ast, self._c_uint64_p):
            return self._c_uint64_p.contents.value
        else:
            return int(z3.Z3_get_numeral_string(ctx, ast))

    def _abstract_fp_val(self, ctx, ast, op_name):
        if op_name == 'FPVal':
            # TODO: do better than this
            fp_mantissa = float(z3.Z3_fpa_get_numeral_significand_string(ctx, ast))
            fp_exp = int(z3.Z3_fpa_get_numeral_exponent_string(ctx, ast, False))
            fp_sign_c = ctypes.c_int()
            z3.Z3_fpa_get_numeral_sign(ctx, ast, ctypes.byref(fp_sign_c))
            fp_sign = -1 if fp_sign_c.value != 0 else 1
            value = fp_sign * fp_mantissa * (2 ** fp_exp)
            return value
        elif op_name == 'MinusZero':
            return -0.0
        elif op_name == 'MinusInf':
            return float('-inf')
        elif op_name == 'PlusZero':
            return 0.0
        elif op_name == 'PlusInf':
            return float('inf')
        elif op_name == 'NaN':
            return float('nan')
        else:
            raise BackendError("Called _abstract_fp_val with unknown type")

    def _abstract_fp_encoded_val(self, ctx, ast):
        decl = z3.Z3_get_app_decl(ctx, ast)
        decl_num = z3.Z3_get_decl_kind(ctx, decl)
        op_name = op_map[z3_op_nums[decl_num]]
        sort = z3.Z3_get_sort(ctx, ast)
        ebits = z3.Z3_fpa_get_ebits(ctx, sort)
        sbits = z3.Z3_fpa_get_sbits(ctx, sort) - 1  # includes sign bit

        if op_name == 'FPVal':
            # TODO: do better than this
            fp_mantissa = int(z3.Z3_fpa_get_numeral_significand_string(ctx, ast))
            fp_exp = int(z3.Z3_fpa_get_numeral_exponent_string(ctx, ast, True))
            fp_sign_c = ctypes.c_int()
            z3.Z3_fpa_get_numeral_sign(ctx, ast, ctypes.byref(fp_sign_c))
            fp_sign = 1 if fp_sign_c.value != 0 else 0
        elif op_name == 'MinusZero':
            fp_sign = 1
            fp_exp = 0
            fp_mantissa = 0
        elif op_name == 'MinusInf':
            fp_sign = 1
            fp_exp = (1<<ebits) - 1
            fp_mantissa = 0
        elif op_name == 'PlusZero':
            fp_sign = 0
            fp_exp = 0
            fp_mantissa = 0
        elif op_name == 'PlusInf':
            fp_sign = 0
            fp_exp = (1<<ebits) - 1
            fp_mantissa = 0
        elif op_name == 'NaN':
            fp_sign = 0
            fp_exp = (1<<ebits) - 1
            fp_mantissa = 1
        else:
            raise BackendError("Called _abstract_fp_val with unknown type")

        value = (fp_sign << (ebits + sbits)) | (fp_exp << sbits) | fp_mantissa
        return value

    def solver(self, timeout=None):
        if not self.reuse_z3_solver or getattr(self._tls, 'solver', None) is None:
            s = z3.Solver(ctx=self._context)
            _add_memory_pressure(1024 * 1024 * 10)
            if self.reuse_z3_solver:
                # Store the Z3 solver to a thread-local storage if the reuse-solver option is enabled
                self._tls.solver = s
        else:
            # Load the existing Z3 solver for this thread
            s = self._tls.solver
            s.reset()

        # for some reason we always reset the solver anyway, so always clear it. REUSE_SOLVER is fundamentally broken
        self._hash_to_constraint.clear()

        # Configure timeouts
        if timeout is not None:
            if 'soft_timeout' in str(s.param_descrs()):
                s.set('soft_timeout', timeout)
                s.set('solver2_timeout', timeout)
            else:
                s.set('timeout', timeout)
        return s

    # @staticmethod
    # def check(solver):
    #     query_smt2 = solver.to_smt2()
    #     start_time = time.time()
    #     result = solver.check()
    #     end_time = time.time()
    #     time_delta = end_time - start_time
    #     BackendZ3.query_record.update(query_smt2, time_delta)
    #     return result

    @staticmethod
    def check(solver):
        query_smt2 = ""
        solver.set('timeout', 1000)
        start_time = time.time()
        result = solver.check()
        end_time = time.time()
        time_delta = end_time - start_time
        if result == z3.unknown:
            query_smt2 = solver.to_smt2()
            solver.set('timeout', 100000)
            start_time = time.time()
            result = solver.check()
            end_time = time.time()
            time_delta = end_time - start_time
        BackendZ3.query_record.update(query_smt2, time_delta)
        return result

    def _add(self, s, c, track=False):
        if track:
            for constraint in c:
                name = str(hash(constraint))
                if name not in self._hash_to_constraint:
                    self._hash_to_constraint[name] = constraint
                    s.assert_and_track(constraint, name)
        else:
            s.add(*c)

    def add(self, s, c, track=False):
        converted = self.convert_list(c)
        if track:
            for a, nice_ast in zip(c, converted):
                ast = nice_ast.ast
                h = self._z3_ast_hash(ast)
                self._ast_cache[h] = (a, ast)
        return self._add(s, converted, track=track)

    def _unsat_core(self, s):
        cores = s.unsat_core()
        constraints = [ ]
        for core in cores:
            constraints.append(self._hash_to_constraint.get(str(core)))
        return constraints

    @condom
    def _primitive_from_model(self, model, expr):
        v = model.eval(expr, model_completion=True)
        return self._abstract_to_primitive(v.ctx.ctx, v.ast)

    #
    # New, model-driven solves
    #

    def _generic_model(self, z3_model):
        """
        Converts a Z3 model to a name->primitive dict.
        """
        model = { }
        for m_f in z3_model:
            n = _z3_decl_name_str(m_f.ctx.ctx, m_f.ast).decode()
            m = m_f()
            me = z3_model.eval(m)
            model[n] = self._abstract_to_primitive(me.ctx.ctx, me.ast)

        return model

    def _satisfiable(self, extra_constraints=(), solver=None, model_callback=None):
        global solve_count

        solve_count += 1
        if len(extra_constraints) > 0:
            solver.push()
            solver.add(*extra_constraints)

        try:

            l.debug("Doing a check!")
            #print "CHECKING"
            if self.check(solver) != z3.sat:
                return False

            if model_callback is not None:
                model_callback(self._generic_model(solver.model()))
        finally:
            if len(extra_constraints) > 0:
                solver.pop()
        return True

    def _eval(self, expr, n, extra_constraints=(), solver=None, model_callback=None):
        results = self._batch_eval(
            [ expr ], n, extra_constraints=extra_constraints,
            solver=solver, model_callback=model_callback
        )

        # Unpack it
        return [ v[0] for v in results ]

    @condom
    def _batch_eval(self, exprs, n, extra_constraints=(), solver=None, model_callback=None):
        global solve_count

        result_values = [ ]

        if len(extra_constraints) > 0 or n != 1:
            solver.push()
        if len(extra_constraints) > 0:
            solver.add(*extra_constraints)

        for i in range(n):
            solve_count += 1
            l.debug("Doing a check!")
            if self.check(solver) != z3.sat:
                break
            model = solver.model()

            # construct results
            r = [ ]
            for expr in exprs:
                if not isinstance(expr, (numbers.Number, str, bool)):
                    v = self._primitive_from_model(model, expr)
                    r.append(v)
                else:
                    r.append(expr)

            # Append the solution to the result list
            if model_callback is not None:
                model_callback(self._generic_model(solver.model()))
            result_values.append(tuple(r))

            # Construct the extra constraint so we don't get the same result anymore
            if i + 1 != n:
                if len(exprs) == 1:
                    solver.add(exprs[0] != r[0])
                else:
                    solver.add(self._op_raw_Not(self._op_raw_And(*[(ex == ex_v) for ex, ex_v in zip(exprs, r)])))
                model = None

        if len(extra_constraints) > 0 or n != 1:
            solver.pop()

        return result_values

    @condom
    def _min(self, expr, extra_constraints=(), solver=None, model_callback=None):
        global solve_count

        lo = 0
        hi = 2**expr.size()-1
        vals = set()

        if len(extra_constraints) > 0:
            solver.push()
            solver.add(*[self.convert(e) for e in extra_constraints])

        numpop = 0

        # TODO: Can only deal with bitvectors, not floats
        while hi-lo > 1:
            middle = (lo + hi)//2
            #l.debug("h/m/l/d: %d %d %d %d", hi, middle, lo, hi-lo)

            solver.push()
            # TODO: is this assumption correct?
            # here it's not safe to call directly the z3 low level API since it might happen that the argument is an
            # integer and not a BV
            solver.add(z3.UGE(expr, lo), z3.ULE(expr, middle))
            numpop += 1

            solve_count += 1
            l.debug("Doing a check!")
            if self.check(solver) == z3.sat:
                l.debug("... still sat")
                if model_callback is not None:
                    model_callback(self._generic_model(solver.model()))
                vals.add(self._primitive_from_model(solver.model(), expr))
                hi = middle
            else:
                l.debug("... now unsat")
                lo = middle
                solver.pop()
                numpop -= 1

        for _ in range(numpop):
            solver.pop()

        #l.debug("final hi/lo: %d, %d", hi, lo)

        if hi == lo:
            vals.add(lo)
        else:
            solver.push()
            solver.add(expr == lo)
            l.debug("Doing a check!")
            if self.check(solver) == z3.sat:
                if model_callback is not None:
                    model_callback(self._generic_model(solver.model()))
                vals.add(lo)
                solver.pop()
            else:
                vals.add(hi)
                solver.pop()

        if len(extra_constraints) > 0:
            solver.pop()

        return min(vals)

    @condom
    def _max(self, expr, extra_constraints=(), solver=None, model_callback=None):
        global solve_count

        lo = 0
        hi = 2**expr.size()-1
        vals = set()

        if len(extra_constraints) > 0:
            solver.push()
            solver.add(*[self.convert(e) for e in extra_constraints])

        numpop = 0

        # TODO: Can only deal with bitvectors, not floats
        while hi-lo > 1:
            middle = (lo + hi)//2
            #l.debug("h/m/l/d: %d %d %d %d", hi, middle, lo, hi-lo)

            solver.push()
            # TODO: is this assumption correct?
            # here it's not safe to call directly the z3 low level API since it might happen that the argument is an
            # integer and not a BV
            solver.add(z3.UGT(expr, middle), z3.ULE(expr, hi))
            numpop += 1

            solve_count += 1
            l.debug("Doing a check!")
            if self.check(solver) == z3.sat:
                l.debug("... still sat")
                lo = middle
                vals.add(self._primitive_from_model(solver.model(), expr))
                if model_callback is not None:
                    model_callback(self._generic_model(solver.model()))
            else:
                l.debug("... now unsat")
                hi = middle
                solver.pop()
                numpop -= 1
            #l.debug("          now: %d %d %d %d", hi, middle, lo, hi-lo)

        for _ in range(numpop):
            solver.pop()

        if hi == lo:
            vals.add(hi)
        else:
            solver.push()
            solver.add(expr == hi)
            l.debug("Doing a check!")
            if self.check(solver) == z3.sat:
                if model_callback is not None:
                    model_callback(self._generic_model(solver.model()))
                vals.add(hi)
                solver.pop()
            else:
                vals.add(lo)
                solver.pop()

        if len(extra_constraints) > 0:
            solver.pop()

        return max(vals)

    def _simplify(self, e): #pylint:disable=W0613,R0201
        raise Exception("This shouldn't be called. Bug Yan.")

    @condom
    def simplify(self, expr):  #pylint:disable=arguments-differ
        if expr._simplified:
            return expr

        if self._enable_simplification_cache:
            try:
                k = self._simplification_cache_key[expr._cache_key]
                #print "HIT WEAK KEY CACHE"
                return k
            except KeyError:
                pass
            try:
                k = self._simplification_cache_val[expr._cache_key]
                #print "HIT WEAK VALUE CACHE"
                return k
            except KeyError:
                pass

            #print "MISS CACHE"

        l.debug("SIMPLIFYING EXPRESSION")

        #print "SIMPLIFYING"

        expr_raw = self.convert(expr)

        #l.debug("... before: %s (%s)", expr_raw, expr_raw.__class__.__name__)

        #s = expr_raw
        if isinstance(expr_raw, z3.BoolRef):
            boolref_tactics = self._boolref_tactics
            s = boolref_tactics(expr_raw).as_expr()
            #n = s.decl().name()
            #if n == 'true':
            #    s = True
            #elif n == 'false':
            #    s = False
        elif isinstance(expr_raw, z3.BitVecRef):
            s = z3.simplify(expr_raw)
        else:
            s = expr_raw

        o = self._abstract(s)
        o._simplified = Base.FULL_SIMPLIFY

        if self._enable_simplification_cache:
            self._simplification_cache_val[expr._cache_key] = o
            self._simplification_cache_key[expr._cache_key] = o
        return o

    def _is_false(self, e, extra_constraints=(), solver=None, model_callback=None):
        return z3.simplify(e).eq(z3.BoolVal(False, ctx=self._context))

    def _is_true(self, e, extra_constraints=(), solver=None, model_callback=None):
        return z3.simplify(e).eq(z3.BoolVal(True, ctx=self._context))

    def _solution(self, expr, v, extra_constraints=(), solver=None, model_callback=None):
        return self._satisfiable(extra_constraints=(expr == v,) + tuple(extra_constraints), solver=solver, model_callback=model_callback)

    #
    # Some Z3 passthroughs
    #

    # these require the context or special treatment

    @staticmethod
    def _op_div(a, b):
        return z3.BitVecRef(z3.Z3_mk_bvudiv(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)
    @staticmethod
    def _op_mod(a, b):
        return z3.BitVecRef(z3.Z3_mk_bvurem(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)
    @staticmethod
    def _op_add(*args):
        return reduce(operator.__add__, args)
    @staticmethod
    def _op_sub(*args):
        return reduce(operator.__sub__, args)
    @staticmethod
    def _op_mul(*args):
        return reduce(operator.__mul__, args)
    @staticmethod
    def _op_or(*args):
        return reduce(operator.__or__, args)
    @staticmethod
    def _op_xor(*args):
        return reduce(operator.__xor__, args)
    @staticmethod
    def _op_and(*args):
        return reduce(operator.__and__, args)

    def _op_raw_And(self, *args):
        # copied from z3._to_ast_array
        sz = len(args)
        _args = (z3.Ast * sz)()
        for i in range(sz):
            _args[i] = args[i].as_ast()
        return z3.BoolRef(z3.Z3_mk_and(self._context.ref(), sz, _args), self._context)

    def _op_raw_Xor(self, *args):
        return z3.BoolRef(z3.Z3_mk_xor(self._context, *(arg.as_ast() for arg in args)), self._context)

    def _op_raw_Or(self, *args):
        # copied from z3._to_ast_array
        sz = len(args)
        _args = (z3.Ast * sz)()
        for i in range(sz):
            _args[i] = args[i].as_ast()
        return z3.BoolRef(z3.Z3_mk_or(self._context.ref(), sz, _args), self._context)

    def _op_raw_Not(self, a):
        return z3.BoolRef(z3.Z3_mk_not(self._context.ref(), a.as_ast()), self._context)

    def _op_raw_If(self, i, t, e):
        # partially copied from z3._to_expr_ref
        ctx_ref = self._context.ref()
        ast = z3.Z3_mk_ite(ctx_ref, i.as_ast(), t.as_ast(), e.as_ast())
        k = z3.Z3_get_ast_kind(ctx_ref, ast)
        sk = z3.Z3_get_sort_kind(ctx_ref, z3.Z3_get_sort(ctx_ref, ast))
        if sk == z3.Z3_BOOL_SORT:
            return z3.BoolRef(ast, self._context)
        if sk == z3.Z3_BV_SORT:
            if k == z3.Z3_NUMERAL_AST:
                return z3.BitVecNumRef(ast, self._context)
            else:
                return z3.BitVecRef(ast, self._context)
        if sk == z3.Z3_FLOATING_POINT_SORT:
            if k == z3.Z3_APP_AST and z3.Z3_is_numeral_ast(ctx_ref, ast):
                return z3.FPNumRef(ast, self._context)
            else:
                return z3.FPRef(ast, self._context)
        return z3.ExprRef(ast, self._context)

    @condom
    def _op_raw_fpAbs(self, a):
        return z3.FPRef(z3.Z3_mk_fpa_abs(self._context.ref(), a.as_ast()), self._context)

    @condom
    def _op_raw_fpNeg(self, a):
        return z3.FPRef(z3.Z3_mk_fpa_neg(self._context.ref(), a.as_ast()), self._context)

    @condom
    def _op_raw_fpAdd(self, rm, a, b):
        return z3.FPRef(z3.Z3_mk_fpa_add(self._context.ref(), rm.as_ast(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpSub(self, rm, a, b):
        return z3.FPRef(z3.Z3_mk_fpa_sub(self._context.ref(), rm.as_ast(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpMul(self, rm, a, b):
        return z3.FPRef(z3.Z3_mk_fpa_mul(self._context.ref(), rm.as_ast(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpDiv(self, rm, a, b):
        return z3.FPRef(z3.Z3_mk_fpa_div(self._context.ref(), rm.as_ast(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpLT(self, a, b):
        return z3.BoolRef(z3.Z3_mk_fpa_lt(self._context.ref(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpLEQ(self, a, b):
        return z3.BoolRef(z3.Z3_mk_fpa_leq(self._context.ref(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpGT(self, a, b):
        return z3.BoolRef(z3.Z3_mk_fpa_gt(self._context.ref(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpGEQ(self, a, b):
        return z3.BoolRef(z3.Z3_mk_fpa_geq(self._context.ref(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpEQ(self, a, b):
        return z3.BoolRef(z3.Z3_mk_fpa_eq(self._context.ref(), a.as_ast(), b.as_ast()), self._context)

    @condom
    def _op_raw_fpIsNaN(self, a):
        return z3.BoolRef(z3.Z3_mk_fpa_is_nan(self._context.ref(), a.as_ast()), self._context)

    @condom
    def _op_raw_fpIsInf(self, a):
        return z3.BoolRef(z3.Z3_mk_fpa_is_inf(self._context.ref(), a.as_ast()), self._context)

    @condom
    def _op_raw_fpFP(self, sgn, exp, sig):
        return z3.FPRef(z3.Z3_mk_fpa_fp(self._context.ref(), sgn.ast, exp.ast, sig.ast), self._context)

    @condom
    def _op_raw_fpToSBV(self, rm, fp, bv_len):
        return z3.BitVecRef(z3.Z3_mk_fpa_to_sbv(self._context.ref(), rm.ast, fp.ast, bv_len), self._context)

    @condom
    def _op_raw_fpToUBV(self, rm, fp, bv_len):
        return z3.BitVecRef(z3.Z3_mk_fpa_to_ubv(self._context.ref(), rm.ast, fp.ast, bv_len), self._context)

    @condom
    def _op_raw_fpToFP(self, a1, a2=None, a3=None):
        # TODO: lots of mandatory checks are performed by the high level API here. we shouldn't use low level APIs here
        return z3.fpToFP(a1, a2=a2, a3=a3, ctx=self._context)

    @condom
    def _op_raw_fpToIEEEBV(self, x):
        return z3.BitVecRef(z3.Z3_mk_fpa_to_ieee_bv(self._context.ref(), x.ast), self._context)

    # and these do not
    @staticmethod
    @condom
    def _op_raw_Concat(*args):
        sz = len(args)
        ctx = None
        for a in args:
            if z3.is_expr(a):
                ctx = a.ctx
                break
        # TODO: I don't think this is needed for us, we don't deal with Seq or Regex
        # if z3.is_seq(args[0]) or isinstance(args[0], str):
        #     v = (z3.Ast * sz)()
        #     for i in range(sz):
        #         v[i] = args[i].as_ast()
        #     return z3.SeqRef(z3.Z3_mk_seq_concat(ctx.ref(), sz, v), ctx)
        #
        # if z3.is_re(args[0]):
        #     v = (z3.Ast * sz)()
        #     for i in range(sz):
        #         v[i] = args[i].as_ast()
        #     return z3.ReRef(z3.Z3_mk_re_concat(ctx.ref(), sz, v), ctx)

        r = args[0]
        for i in range(sz - 1):
            r = z3.BitVecRef(z3.Z3_mk_concat(ctx.ref(), r.as_ast(), args[i + 1].as_ast()), ctx)
        return r

    @staticmethod
    @condom
    def _op_raw_Extract(high, low, a):
        # TODO: I don't think this is needed for us, we don't deal with Seq or Regex
        # if isinstance(high, str):
        #     high = z3.StringVal(high)
        # if z3.is_seq(high):
        #     s = high
        #     offset, length = _coerce_exprs(low, a, s.ctx)
        #     return z3.SeqRef(z3.Z3_mk_seq_extract(s.ctx_ref(), s.as_ast(), offset.as_ast(), length.as_ast()), s.ctx)
        return z3.BitVecRef(z3.Z3_mk_extract(a.ctx_ref(), high, low, a.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_LShR(a, b):
        return z3.BitVecRef(z3.Z3_mk_bvlshr(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_RotateLeft(a, b):
        return z3.BitVecRef(z3.Z3_mk_ext_rotate_left(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_RotateRight(a, b):
        return z3.BitVecRef(z3.Z3_mk_ext_rotate_right(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_SignExt(n, a):
        return z3.BitVecRef(z3.Z3_mk_sign_ext(a.ctx_ref(), n, a.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_UGE(a, b):
        return z3.BoolRef(z3.Z3_mk_bvuge(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_UGT(a, b):
        return z3.BoolRef(z3.Z3_mk_bvugt(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_ULE(a, b):
        return z3.BoolRef(z3.Z3_mk_bvule(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_ULT(a, b):
        return z3.BoolRef(z3.Z3_mk_bvult(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_ZeroExt(n, a):
        return z3.BitVecRef(z3.Z3_mk_zero_ext(a.ctx_ref(), n, a.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_SMod(a, b):
        return z3.BitVecRef(z3.Z3_mk_bvsrem(a.ctx_ref(), a.as_ast(), b.as_ast()), a.ctx)

    @staticmethod
    @condom
    def _op_raw_Reverse(a):
        if a.size() == 8:
            return a
        elif a.size() % 8 != 0:
            raise ClaripyOperationError("can't reverse non-byte sized bitvectors")
        else:
            return BackendZ3._op_raw_Concat(*[BackendZ3._op_raw_Extract(i+7, i, a) for i in range(0, a.size(), 8)])

    @staticmethod
    @condom
    def _op_raw_SLT(a, b):
        return a < b

    @staticmethod
    @condom
    def _op_raw_SLE(a, b):
        return a <= b

    @staticmethod
    @condom
    def _op_raw_SGT(a, b):
        return a > b

    @staticmethod
    @condom
    def _op_raw_SGE(a, b):
        return a >= b

    @staticmethod
    @condom
    def _op_raw_SDiv(a, b):
        return a / b

    def _identical(self, a, b):
        return a.eq(b)

#
# this is for the actual->abstract conversion above
#

# these are Z3 operation names for abstraction
z3_op_names = [ _ for _ in dir(z3) if _.startswith('Z3_OP') ]
z3_op_nums = { getattr(z3, _): _ for _ in z3_op_names }

#pylint:disable=bad-continuation
op_map = {
    # Boolean
    'Z3_OP_TRUE': 'True',
    'Z3_OP_FALSE': 'False',
    'Z3_OP_EQ': '__eq__',
    'Z3_OP_DISTINCT': '__ne__',
    'Z3_OP_ITE': 'If',
    'Z3_OP_AND': 'And',
    'Z3_OP_OR': 'Or',
    'Z3_OP_IFF': '__eq__',
    'Z3_OP_XOR': 'Xor',
    'Z3_OP_NOT': 'Not',
    'Z3_OP_IMPLIES': 'Implies',
    #'Z3_OP_OEQ': None,

    # Arithmetic
    #'Z3_OP_ANUM': None,
    #'Z3_OP_AGNUM': None,
    'Z3_OP_LE': 'SLE',
    'Z3_OP_GE': 'SGE',
    'Z3_OP_LT': 'SLT',
    'Z3_OP_GT': 'SGT',
    'Z3_OP_ADD': '__add__',
    'Z3_OP_SUB': '__sub__',
    'Z3_OP_UMINUS': '__neg__',
    'Z3_OP_MUL': '__mul__',
    'Z3_OP_DIV': 'SDiv',
    'Z3_OP_IDIV': 'SDiv',
    'Z3_OP_REM': '__mod__',
    'Z3_OP_MOD': '__mod__',
    #'Z3_OP_TO_REAL': None,
    #'Z3_OP_TO_INT': None,
    #'Z3_OP_IS_INT': None,
    'Z3_OP_POWER': '__pow__',

    # Arrays & Sets
    #'Z3_OP_STORE': None,
    #'Z3_OP_SELECT': None,
    #'Z3_OP_CONST_ARRAY': None,
    #'Z3_OP_ARRAY_MAP': None,
    #'Z3_OP_ARRAY_DEFAULT': None,
    #'Z3_OP_SET_UNION': None,
    #'Z3_OP_SET_INTERSECT': None,
    #'Z3_OP_SET_DIFFERENCE': None,
    #'Z3_OP_SET_COMPLEMENT': None,
    #'Z3_OP_SET_SUBSET': None,
    #'Z3_OP_AS_ARRAY': None,

    # Bit-vectors
    'Z3_OP_BNUM': 'BitVecVal',
    #'Z3_OP_BIT1': None, # MAYBE TODO
    #'Z3_OP_BIT0': None, # MAYBE TODO
    'Z3_OP_BNEG': '__neg__',
    'Z3_OP_BADD': '__add__',
    'Z3_OP_BSUB': '__sub__',
    'Z3_OP_BMUL': '__mul__',

    'Z3_OP_BSDIV': 'SDiv',
    'Z3_OP_BUDIV': '__floordiv__',
    'Z3_OP_BSREM': 'SMod',
    'Z3_OP_BUREM': '__mod__',
    'Z3_OP_BSMOD': 'SMod',
    'Z3_OP_BSDIV_I': 'SDiv',
    'Z3_OP_BUDIV_I': '__floordiv__',
    'Z3_OP_BSREM_I': 'SMod',
    'Z3_OP_BUREM_I': '__mod__',
    'Z3_OP_BSMOD_I': 'SMod',

    # special functions to record the division by 0 cases
    # these are internal functions
    #'Z3_OP_BSDIV0': None,
    #'Z3_OP_BUDIV0': None,
    #'Z3_OP_BSREM0': None,
    #'Z3_OP_BUREM0': None,
    #'Z3_OP_BSMOD0': None,

    'Z3_OP_ULEQ': 'ULE',
    'Z3_OP_SLEQ': 'SLE',
    'Z3_OP_UGEQ': 'UGE',
    'Z3_OP_SGEQ': 'SGE',
    'Z3_OP_ULT': 'ULT',
    'Z3_OP_SLT': 'SLT',
    'Z3_OP_UGT': 'UGT',
    'Z3_OP_SGT': 'SGT',

    'Z3_OP_BAND': '__and__',
    'Z3_OP_BOR': '__or__',
    'Z3_OP_BNOT': '__invert__',
    'Z3_OP_BXOR': '__xor__',
    #'Z3_OP_BNAND': None,
    #'Z3_OP_BNOR': None,
    #'Z3_OP_BXNOR': None,

    'Z3_OP_CONCAT': 'Concat',
    'Z3_OP_SIGN_EXT': 'SignExt',
    'Z3_OP_ZERO_EXT': 'ZeroExt',
    'Z3_OP_EXTRACT': 'Extract',
    'Z3_OP_REPEAT': 'RepeatBitVec',

    #'Z3_OP_BREDOR': None,
    #'Z3_OP_BREDAND': None,
    #'Z3_OP_BCOMP': None,

    'Z3_OP_BSHL': '__lshift__',
    'Z3_OP_BLSHR': 'LShR',
    'Z3_OP_BASHR': '__rshift__',
    #'Z3_OP_ROTATE_LEFT': None,
    #'Z3_OP_ROTATE_RIGHT': None,
    'Z3_OP_EXT_ROTATE_LEFT': 'RotateLeft',
    'Z3_OP_EXT_ROTATE_RIGHT': 'RotateRight',

    'Z3_OP_FPA_TO_SBV': 'fpToSBV',
    'Z3_OP_FPA_TO_UBV': 'fpToUBV',
    'Z3_OP_FPA_TO_IEEE_BV': 'fpToIEEEBV',
    'Z3_OP_FPA_TO_FP': 'fpToFP',
    'Z3_OP_FPA_NUM': 'FPVal',

    'Z3_OP_FPA_MINUS_ZERO': 'MinusZero',
    'Z3_OP_FPA_MINUS_INF': 'MinusInf',
    'Z3_OP_FPA_PLUS_ZERO': 'PlusZero',
    'Z3_OP_FPA_PLUS_INF': 'PlusInf',
    'Z3_OP_FPA_NAN': 'NaN',

    'Z3_OP_FPA_EQ': 'fpEQ',
    'Z3_OP_FPA_GT': 'fpGT',
    'Z3_OP_FPA_GE': 'fpGEQ',
    'Z3_OP_FPA_LT': 'fpLT',
    'Z3_OP_FPA_LE': 'fpLEQ',

    'Z3_OP_FPA_ABS': 'fpAbs',
    'Z3_OP_FPA_NEG': 'fpNeg',
    'Z3_OP_FPA_ADD': 'fpAdd',
    'Z3_OP_FPA_SUB': 'fpSub',
    'Z3_OP_FPA_MUL': 'fpMul',
    'Z3_OP_FPA_DIV': 'fpDiv',

    'Z3_OP_FPA_RM_NEAREST_TIES_TO_EVEN': 'RM_RNE',
    'Z3_OP_FPA_RM_NEAREST_TIES_TO_AWAY': 'RM_RNA',
    'Z3_OP_FPA_RM_TOWARD_ZERO': 'RM_RTZ',
    'Z3_OP_FPA_RM_TOWARD_POSITIVE': 'RM_RTP',
    'Z3_OP_FPA_RM_TOWARD_NEGATIVE': 'RM_RTN',

    'Z3_OP_UNINTERPRETED': 'UNINTERPRETED'
}

from ..ast.base import Base
from ..ast.bv import BV, BVV
from ..ast.bool import BoolV, Bool
from ..ast.fp import FP, FPV
from ..ast.strings import StringV, StringS
from ..operations import backend_operations, backend_fp_operations
from ..fp import FSort, RM, RM_NearestTiesEven, RM_NearestTiesAwayFromZero, RM_TowardsPositiveInf, RM_TowardsNegativeInf, RM_TowardsZero
from ..errors import ClaripyError, BackendError, ClaripyOperationError
from .. import _all_operations

op_type_map = {
    # Boolean
    'Z3_OP_TRUE': Bool,
    'Z3_OP_FALSE': Bool,
    'Z3_OP_EQ': Bool,
    'Z3_OP_DISTINCT': Bool,
    'Z3_OP_ITE': Bool,
    'Z3_OP_AND': Bool,
    'Z3_OP_OR': Bool,
    'Z3_OP_IFF': Bool,
    'Z3_OP_XOR': Bool,
    'Z3_OP_NOT': Bool,
    'Z3_OP_IMPLIES': Bool,
    #'Z3_OP_OEQ': None,

    # Arithmetic
    #'Z3_OP_ANUM': None,
    #'Z3_OP_AGNUM': None,
    'Z3_OP_LE': None,
    'Z3_OP_GE': None,
    'Z3_OP_LT': None,
    'Z3_OP_GT': None,
    'Z3_OP_ADD': None,
    'Z3_OP_SUB': None,
    'Z3_OP_UMINUS': None,
    'Z3_OP_MUL': None,
    'Z3_OP_DIV': None,
    'Z3_OP_IDIV': None,
    'Z3_OP_REM': None, # TODO: is this correct?
    'Z3_OP_MOD': None,
    #'Z3_OP_TO_REAL': None,
    #'Z3_OP_TO_INT': None,
    #'Z3_OP_IS_INT': None,
    'Z3_OP_POWER': None,

    # Arrays & Sets
    #'Z3_OP_STORE': None,
    #'Z3_OP_SELECT': None,
    #'Z3_OP_CONST_ARRAY': None,
    #'Z3_OP_ARRAY_MAP': None,
    #'Z3_OP_ARRAY_DEFAULT': None,
    #'Z3_OP_SET_UNION': None,
    #'Z3_OP_SET_INTERSECT': None,
    #'Z3_OP_SET_DIFFERENCE': None,
    #'Z3_OP_SET_COMPLEMENT': None,
    #'Z3_OP_SET_SUBSET': None,
    #'Z3_OP_AS_ARRAY': None,

    # Bit-vectors
    'Z3_OP_BNUM': 'BitVecVal',
    #'Z3_OP_BIT1': None, # MAYBE TODO
    #'Z3_OP_BIT0': None, # MAYBE TODO
    'Z3_OP_BNEG': BV,
    'Z3_OP_BADD': BV,
    'Z3_OP_BSUB': BV,
    'Z3_OP_BMUL': BV,

    'Z3_OP_BSDIV': BV,
    'Z3_OP_BUDIV': BV,
    'Z3_OP_BSREM': BV,
    'Z3_OP_BUREM': BV,
    'Z3_OP_BSMOD': BV,
    'Z3_OP_BSDIV_I': BV,
    'Z3_OP_BUDIV_I': BV,
    'Z3_OP_BSREM_I': BV,
    'Z3_OP_BUREM_I': BV,
    'Z3_OP_BSMOD_I': BV,

    # special functions to record the division by 0 cases
    # these are internal functions
    #'Z3_OP_BSDIV0': None,
    #'Z3_OP_BUDIV0': None,
    #'Z3_OP_BSREM0': None,
    #'Z3_OP_BUREM0': None,
    #'Z3_OP_BSMOD0': None,

    'Z3_OP_ULEQ': Bool,
    'Z3_OP_SLEQ': Bool,
    'Z3_OP_UGEQ': Bool,
    'Z3_OP_SGEQ': Bool,
    'Z3_OP_ULT': Bool,
    'Z3_OP_SLT': Bool,
    'Z3_OP_UGT': Bool,
    'Z3_OP_SGT': Bool,

    'Z3_OP_BAND': BV,
    'Z3_OP_BOR': BV,
    'Z3_OP_BNOT': BV,
    'Z3_OP_BXOR': BV,
    #'Z3_OP_BNAND': None,
    #'Z3_OP_BNOR': None,
    #'Z3_OP_BXNOR': None,

    'Z3_OP_CONCAT': BV,
    'Z3_OP_SIGN_EXT': BV,
    'Z3_OP_ZERO_EXT': BV,
    'Z3_OP_EXTRACT': BV,
    'Z3_OP_REPEAT': BV,

    #'Z3_OP_BREDOR': None,
    #'Z3_OP_BREDAND': None,
    #'Z3_OP_BCOMP': None,

    'Z3_OP_BSHL': BV,
    'Z3_OP_BLSHR': BV,
    'Z3_OP_BASHR': BV,
    #'Z3_OP_ROTATE_LEFT': None,
    #'Z3_OP_ROTATE_RIGHT': None,
    'Z3_OP_EXT_ROTATE_LEFT': BV,
    'Z3_OP_EXT_ROTATE_RIGHT': BV,

    'Z3_OP_FPA_TO_SBV': BV,
    'Z3_OP_FPA_TO_UBV': BV,
    'Z3_OP_FPA_TO_IEEE_BV': BV,
    'Z3_OP_FPA_TO_FP': FP,
    'Z3_OP_FPA_NUM': FP,

    'Z3_OP_FPA_MINUS_ZERO': FP,
    'Z3_OP_FPA_MINUS_INF': FP,
    'Z3_OP_FPA_PLUS_ZERO': FP,
    'Z3_OP_FPA_PLUS_INF': FP,
    'Z3_OP_FPA_NAN': FP,

    'Z3_OP_FPA_EQ': Bool,
    'Z3_OP_FPA_GT': Bool,
    'Z3_OP_FPA_GE': Bool,
    'Z3_OP_FPA_LT': Bool,
    'Z3_OP_FPA_LE': Bool,

    'Z3_OP_FPA_ABS': FP,
    'Z3_OP_FPA_NEG': FP,
    'Z3_OP_FPA_ADD': FP,
    'Z3_OP_FPA_SUB': FP,
    'Z3_OP_FPA_MUL': FP,
    'Z3_OP_FPA_DIV': FP,

    'Z3_OP_UNINTERPRETED': 'UNINTERPRETED'
}
