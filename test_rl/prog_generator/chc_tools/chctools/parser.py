import pysmt
import sys

import functools
import collections

from pysmt.smtlib.parser import SmtLibZ3Parser, SmtLibCommand
from pysmt.exceptions import UnknownSmtLibCommandError, PysmtValueError, PysmtSyntaxError

import pysmt.operators as op
import pysmt.typing as types

Rule = collections.namedtuple('Rule', ['formula', 'is_query'])


class ChcRulesSmtLibParser(SmtLibZ3Parser):
    def __init__(self, env=None, interactive=False):
        super().__init__(env, interactive)

        # Add new commands
        self.commands["set-logic"] = self._cmd_set_logic
        self.commands["declare-rel"] = self._cmd_declare_rel
        self.commands["declare-var"] = self._cmd_declare_var
        self.commands["rule"] = self._cmd_rule
        self.commands["query"] = self._cmd_query

        # Remove unused commands
        del self.commands["get-value"]
        # ...

        self.interpreted["div"] = self._operator_adapter(self._division)

        # some interpreted functions in Z3 are not supported
        # by pysmt. Map them to uninterpreted functions for now
        # declare mod
        int_sort = self.env.type_manager.INT()
        mod_sort = self.env.type_manager.FunctionType(int_sort, [int_sort, int_sort])
        mod_fn = self._get_var("mod", mod_sort)
        self.cache.bind("mod", \
                        functools.partial(self._function_call_helper, mod_fn))
        # delcare rem
        rem_fn = self._get_var("rem", mod_sort)
        self.cache.bind("rem", \
                        functools.partial(self._function_call_helper, rem_fn))


    def _division(self, left, right):
        """Utility function that builds a division"""
        mgr = self.env.formula_manager
        if left.is_constant() and right.is_constant():
            return mgr.Real(Fraction(left.constant_value()) / \
                            Fraction(right.constant_value()))

        # for some reason pysmt does not like integer division
        if right.is_constant(types.INT):
            return mgr.create_node(node_type=op.DIV, args=(left, right))

        return mgr.Div(left, right)


    def _cmd_set_logic(self, current, tokens):
        elements = self.parse_atoms(tokens, current, 1)
        name = elements[0]
        return SmtLibCommand(current, [None])

    def _cmd_declare_rel(self, current, tokens):
        """(declare-rel <symbol> (<sort>*))"""
        rel = self.parse_atom(tokens, current)
        args_sort = self.parse_params(tokens, current)
        self.consume_closing(tokens, current)

        fn_sort = self.env.type_manager.BOOL()

        if args_sort:
            fn_sort = self.env.type_manager.FunctionType(fn_sort, args_sort)

        fn = self._get_var(rel, fn_sort)
        if fn.symbol_type().is_function_type():
            self.cache.bind(rel, \
                            functools.partial(self._function_call_helper, fn))
        else:
            self.cache.bind(rel, fn)
        return SmtLibCommand(current, [fn])
    def _cmd_declare_var(self, current, tokens):
        """(declare-var <symbol> <sort>)"""
        var = self.parse_atom(tokens, current)
        typename = self.parse_type(tokens, current)
        self.consume_closing(tokens, current)
        v = self._get_var(var, typename)
        self.cache.bind(var, v)
        return SmtLibCommand(current, [v])
    def _cmd_rule(self, current, tokens):
        # print(current)
        expr = self.get_expression(tokens)
        self.consume_closing(tokens, current)
        return Rule(expr, False)

    def _cmd_query(self, current, tokens):
        expr = self.get_expression(tokens)
        self.consume_closing(tokens, current)
        return Rule(expr, True)

    def get_chc(self, script):
        rules = []
        queries = []
        init = self.env.formula_manager.TRUE()
        trans = self.env.formula_manager.TRUE()

        for cmd in self.get_command_generator(script):
            # Simply skip declarations and other commands...
            if type(cmd) == Rule:
                if cmd.is_query:
                    queries.append(cmd.formula)
                else:
                    rules.append(cmd.formula)

        return rules, queries 

def main():
    with open(sys.argv[1], 'r') as script:
        parser = ChcRulesSmtLibParser()
        try:
            r, q = parser.get_chc(script)
            print(r)
            print(q)
        except PysmtSyntaxError as e:
            print(e)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
