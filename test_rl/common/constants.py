import re

AST_EDGE_TYPE = 0
CONTROL_EDGE_TYPE = 2
VAR_LINK_TYPE = 4

NUM_EDGE_TYPES = 6 # 3 edge types x 2 directions


# boogie results
AC_CODE = 0
POST_FAIL_CODE = 1
INDUCTIVE_FAIL_CODE = 2
ENTRY_FAIL_CODE = 3
INVALID_CODE = 4

# z3 pre-check
ALWAYS_TRUE_EXPR_CODE = 3
ALWAYS_FALSE_EXPR_CODE = 4
NORMAL_EXPR_CODE = 5

from code2inv.common.cmd_args import cmd_args

Z3_CMP = ("<", ">", "<=", ">=", "==")
Z3_OP = ("+", "-", "/", "*")
MAX_CHILD = 10
if cmd_args.inv_grammar is None:
    LIST_PREDICATES = [w for w in cmd_args.list_pred.split(',')]
    LIST_OP = [w for w in cmd_args.list_op.split(',')]

    LIST_VAR = None

    if cmd_args.invar_vars is not None:
        LIST_VAR = [w for w in cmd_args.invar_vars.split(',')]
    RULESET = None
else:
    MAX_CHILD = 0
    with open(cmd_args.inv_grammar, 'r') as grammarfile:
        rules = []
        op_mapping = {}
        RULESET = {}
        for line in grammarfile.readlines():
            r_list = line.split("::=")
            if len(r_list) == 2:
                rules.append(r_list)
            else:
                assert len(r_list) == 1
                mapping = r_list[0].split(":")
                if len(mapping) == 2:
                    # print("MAP", mapping)
                    op_mapping[mapping[0].split()[0]] = mapping[1].split()[0]
        for rule in rules:
            assert len(rule) == 2
            rule_lhs = rule[0].split()[0]
            data = rule[1]
            PATTERN = re.compile(r'''((?:[^\|"']|"[^"]*"|'[^']*')+)''')
            rule_rhs = []



            for rule_exp in PATTERN.split(data)[1:-1:2]:
                rule_exp_list = []
                for el in rule_exp.split():
                    if el == "'||'" or el == "\"||\"":
                        rule_exp_list.append("||")
                    else:
                        rule_exp_list.append(el)
                
                if len(rule_exp_list) > MAX_CHILD:
                    MAX_CHILD = len(rule_exp_list)
                rule_rhs.append(rule_exp_list)
                
            for i in range(len(rule_rhs)):
                if len(rule_rhs[i]) == 1 and rule_rhs[i][0] in op_mapping:
                    rule_rhs[i][0] = op_mapping[rule_rhs[i][0]]
            RULESET[rule_lhs] = rule_rhs
        print(RULESET)

print("MAX CHILD", MAX_CHILD)
MAX_DEPTH = 10
if cmd_args.var_format is None:
    VAR_FORMAT = None
elif cmd_args.var_format == "":
    VAR_FORMAT = None
else:
    VAR_FORMAT = cmd_args.var_format

print("VAR FORMAT", VAR_FORMAT)
if cmd_args.single_sample is not None:
    print("SINGLE SAMPLE", cmd_args.single_sample)