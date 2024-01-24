import json
import sys

from code2inv.common.cmd_args import cmd_args
from code2inv.common.constants import AST_EDGE_TYPE, CONTROL_EDGE_TYPE, VAR_LINK_TYPE, Z3_OP, Z3_CMP, VAR_FORMAT
from code2inv.common.checker import z3_check_implication

class GraphNode(object):
    def __init__(self, index, node_type, name = None):
        self.index = index
        self.node_type = node_type
        
        self.name = name

        self.in_edge_list = []
        self.out_edge_list = []

    def add_in_edge(self, src, edge_type):
        self.in_edge_list.append((edge_type, src))

    def add_out_edge(self, dst, edge_type):
        self.out_edge_list.append((edge_type, dst))

    def __repr__(self):
        if self.name is not None and self.node_type is not None:
            return self.name + ", " + self.node_type
        else:
            return "NA"

class ExprNode(object):
    def __init__(self, pg_node):
        self.pg_node = None
        if type(pg_node) is not GraphNode:
            self.name = pg_node
        else:
            self.pg_node = pg_node
            self.name = pg_node.name
        self.children = []
        self.state = None

    def clone(self):
        if self.pg_node is None:
            root = ExprNode(self.name)
        else:
            root = ExprNode(self.pg_node)
        
        for c in self.children:
            root.children.append(c.clone())
        
        return root
    
    def depth(self, d = 0):
        if self.children is None or len(self.children) == 0:
            return d
        else:
            return max([ child.depth(d + 1) for child in self.children ])

    def __str__(self):
        child_list = self.children
        st = ''
        if self.name == '&&' or self.name == '||':
            if len(child_list) == 1:
                return child_list[0].__str__()
            elif len(child_list) == 0:
                return self.name
            else:
                for i in range(len(child_list)):
                    st += '( ' + child_list[i].__str__() + ' )'
                    # st += child_list[i].__str__()
                    if i + 1 < len(child_list):
                        st += ' %s ' % self.name
        else:
            assert len(child_list) == 0 or len(child_list) == 2
            if len(child_list):
                st = child_list[0].__str__() + self.name + child_list[1].__str__()
            else:
                st = self.name
        return st
    
    def to_eval_str(self):
        child_list = self.children
        st = ''
        if self.name == '&&' or self.name == '||':
            if len(child_list) == 1:
                return child_list[0].__str__()
            elif len(child_list) == 0:
                if self.name == "&&":
                    return "and"
                elif self.name == "||":
                    return "or"
                else:
                    return self.name
            else:
                for i in range(len(child_list)):
                    st += '( ' + child_list[i].__str__() + ' )'
                    # st += child_list[i].__str__()
                    if i + 1 < len(child_list):
                        if self.name == "&&":
                            st += ' and '
                        elif self.name == "||":
                            st += ' or '
                        else:
                            st += ' %s ' % self.name
        else:
            assert len(child_list) == 0 or len(child_list) == 2
            if len(child_list):
                st = child_list[0].__str__() + self.name + child_list[1].__str__()
            else:
                st = self.name
        return st

    def __repr__(self):
        child_list = self.children
        st = ''
        if self.name == '&&' or self.name == '||':
            if len(child_list) == 1:
                return child_list[0].__str__()
            elif len(child_list) == 0:
                return self.name
            else:
                for i in range(len(child_list)):
                    st += '( ' + child_list[i].__str__() + ' )'
                    # st += child_list[i].__str__()
                    if i + 1 < len(child_list):
                        st += ' %s ' % self.name
        else:
            assert len(child_list) == 0 or len(child_list) == 2
            if len(child_list):
                st = child_list[0].__str__() + self.name + child_list[1].__str__()
            else:
                st = self.name
        return st

    def __eq__(self, othernode):
        if othernode is None:
            return False
        res = (self.name == othernode.name) and (self.pg_node == othernode.pg_node) and (self.children == othernode.children)
        return res

    def to_smt2(self):
        if len(self.children) > 0:
            op = self.name
            if op == "&&":
                op = "and"
            elif op == "||":
                op = "or"
            elif op == "==":
                op = "="
            elif op == "":
                if len(self.children) == 3 and self.children[1].name in ("<" , ">" , "==" , "<=" , ">=", "+", "-"):
                    return "(%s %s)" % (self.children[1].name, " ".join([self.children[0].to_smt2(), self.children[2].to_smt2()]))
            return "(%s %s)" % (op, " ".join( [c.to_smt2() for c in self.children] ))
        return self.name 

    def has_internal_implications(self,pg):
        child_list = self.children
        if self.name == "&&" or self.name == "||":
            z3_str = [ x.to_z3() for x in child_list ]
            N = len(child_list)
            for i in range(N-1):
                for j in range(i+1,N):
                    if z3_check_implication(pg, z3_str[i], z3_str[j]):
                        # print(z3_str[i], z3_str[j])
                        #print("internal implication check worked", self.to_py)
                        return True
        
        if self.name in ("&&", "||"):
            for x in child_list:
                if x.has_internal_implications(pg):
                    return True

        return False 



    def has_trivial_pattern(self):
        if self.name == "-":
            return str(self.children[0]) == str(self.children[1])        
        elif self.children:
            for x in self.children:
                if x is not None and x.has_trivial_pattern():
                    return True
            
        return False

    def is_z3_boolean(self):
        if self.name in ("&&", "||"):
            for child in self.children:
                if not child.is_z3_boolean():
                    return False
            return True
        elif self.name in Z3_CMP:
            return True
        else:
            return False

    def to_z3(self):
        child_list = self.children
        if self.name == "&&":
            return "z3.And(" + ','.join( [ x.to_z3() for x in child_list ] ) + ")"
        elif self.name == "||":
            return "z3.Or(" + ','.join( [ x.to_z3() for x in child_list ] ) + ")"
        elif len(child_list) == 2:
            return  "(%s %s %s)" % (child_list[0].to_z3(), self.name, child_list[1].to_z3() )
        else:
            assert len(child_list) == 0
            return self.name

    def get_vars(self, st):
        child_list = self.children
        if len(child_list) == 0:
            if self.name.startswith('-') or self.name.isdigit() or self.name in ("<" , ">" , "==" , "<=" , ">=", "+", "-"):
                pass
            else:
                st.add(self.name)
        else:
            for x in child_list:
                x.get_vars(st)


    def to_py(self):
        child_list = self.children
        if self.name == "&&":
            r = " and ".join( [ x.to_py() for x in child_list] )
            return "( " + r + " )"
            #return "z3.And(" + ','.join( [ x.to_z3() for x in self.children] ) + ")"
        elif self.name == "||":
            r = " or ".join( [ x.to_py() for x in child_list] )
            return "( " + r + " )"
            #return "z3.Or(" + ','.join( [ x.to_z3() for x in self.children] ) + ")"
        elif len(child_list) == 2:
            return "( " + child_list[0].to_py() + " " + self.name + " " + child_list[1].to_py() + " )"
            #return  "(%s %s %s)" % (self.children[0].to_z3(), self.name, self.children[1].to_z3() )
        else:
            if self.name == "" and len(self.children) == 3 and self.children[1].name in ("<" , ">" , "==" , "<=" , ">=", "+", "-"):
                return "( " + self.children[0].to_py() + " " + self.children[1].name + " " + self.children[2].to_py() + " )"
            assert len(child_list) == 0
            return self.name

class ProgramGraph(object):
    def __init__(self, graph_json):        
        # raise Exception("Some Exception")
        self.node_list = []

        self.semantic_nodes = {}
        self.raw_variable_nodes = {}
        self.ssa_variable_nodes = {}
        self.const_nodes = {}

        self.unique_nodes = {}
        self.edge_list = []
        
        self.add_raw_variables(graph_json['nodes'])
        self.add_const_values(graph_json['nodes'])
        self.core_vars = list(self.raw_variable_nodes)
        if cmd_args.encoder_model != "Param":
            if VAR_FORMAT == "ssa":
                self.add_ssa_variables(graph_json['nodes'])
            self.unk_node = self.add_node('unk', 'unk_val')

            self.assert_statement = None
            self.if_before_assert = None

            assert_node = None
            for key in graph_json['nodes']:            
                val = graph_json['nodes'][key]            
                sub_root = self.add_node(node_type='internal', name=key)
                self.semantic_nodes[key] = sub_root
                self.traverse_ast(sub_root, val)
                
                if 'cmd' in val and val['cmd'] == 'Assert':                
                    assert_node = key
                    
            pre_true_if = None
            for edges in graph_json['control-flow']:
                assert edges[0] in self.semantic_nodes and edges[1] in self.semantic_nodes
                x = self.semantic_nodes[edges[0]].index
                y = self.semantic_nodes[edges[1]].index
                
                if edges[1] == assert_node:
                    if 'cmd' in graph_json['nodes'][edges[0]]:
                        cmd_x = graph_json['nodes'][edges[0]]['cmd']
                        if cmd_x == 'TrueBranch':
                            pre_true_if = edges[0]

                self.add_double_dir_edge(x, y, CONTROL_EDGE_TYPE, CONTROL_EDGE_TYPE + 1)
                
            pre_if = None
            if pre_true_if:
                for edges in graph_json['control-flow']:
                    if edges[1] == pre_true_if:
                        if 'cmd' in graph_json['nodes'][edges[0]]:
                            cmd_x = graph_json['nodes'][edges[0]]['cmd']
                            if cmd_x == 'if':
                                pre_if = edges[0]
                assert pre_if is not None

            assert assert_node is not None

            self.core_vars = set()        
            if pre_if is not None:
                val = graph_json['nodes'][pre_if]
                self.if_before_assert = self.build_assert(val['rval'], self.core_vars)
            val = graph_json['nodes'][assert_node]
            self.assert_statement = self.build_assert(val['rval'], self.core_vars)

            assert self.assert_statement is not None and len(self.core_vars)
            self.core_var_indices = []
            for key in self.core_vars:
                var_node = self.raw_variable_nodes[key]
                self.core_var_indices.append( var_node.index )
            self.core_var_indices.sort()

        # print("CONST NODES", self.const_nodes)
        # print("VAR NODES", self.raw_variable_nodes)
        # print("SSA VAR NODES", self.ssa_variable_nodes)

    def build_assert(self, d, core_vars):
        # print(d)
        if 'OP' in d:
            node = ExprNode(d['OP'])
            for key in d:
                if key != 'OP':
                    node.children.append( self.build_assert(d[key], core_vars) )
            assert len(node.children) == 2 # binary operator
        elif 'cmd' in d:
            node = self.build_assert(d['rval'], core_vars)
        else:
            assert len(d) == 1
            if 'Var' in d:
                if VAR_FORMAT == "ssa":
                    name = '_'.join(d['Var'].split('_')[0:-1])
                else:
                    name = d['Var']
                core_vars.add(name)
                node = self.raw_variable_nodes[name]
            else:
                assert 'Const' in d
                name = d['Const']
                node = self.const_nodes[name]
            node = ExprNode(node)
        return node

    def add_ssa_variables(self, d):
        if type(d) is not dict:
            return
        if 'Var' in d:
            assert len(d) == 1 # variable node
            if d['Var'] in self.ssa_variable_nodes:
                return
            var_name = '_'.join(d['Var'].split('_')[0:-1])
            assert var_name in self.raw_variable_nodes
            var_node = self.raw_variable_nodes[var_name]
            # print(self.ssa_variable_nodes)
            ssa_node = self.add_node('ssa_variable', d['Var'])
            self.ssa_variable_nodes[d['Var']] = ssa_node
            self.add_double_dir_edge(ssa_node.index, var_node.index, VAR_LINK_TYPE, VAR_LINK_TYPE + 1)
        else:
            for key in d:
                self.add_ssa_variables(d[key])

    def add_raw_variables(self, d):
        if type(d) is not dict:
            return
        if 'Var' in d:
            assert len(d) == 1 # variable node
            if VAR_FORMAT == "ssa":
                var_name = '_'.join(d['Var'].split('_')[0:-1])
            else:
                var_name = d['Var']
            if not var_name in self.raw_variable_nodes:
                self.raw_variable_nodes[var_name] = self.add_node('raw_variable', var_name)
        else:
            for key in d:
                self.add_raw_variables(d[key])

    def add_const_values(self, d):
        if type(d) is not dict:
            return
        if 'Const' in d:
            assert len(d) == 1 # variable node
            const_val = d['Const']            
            if not const_val in self.const_nodes:
                self.const_nodes[const_val] = self.add_node('const', const_val)
        else:
            for key in d:
                self.add_const_values(d[key])

    def traverse_ast(self, node, d):        
        if 'Var' in d: # var node
            assert len(d) == 1 # variable node            
            if VAR_FORMAT == "ssa":
                var_node = self.ssa_variable_nodes[d['Var']]
            else:
                var_node = self.raw_variable_nodes[d['Var']]
            self.add_double_dir_edge(node.index, var_node.index, AST_EDGE_TYPE, AST_EDGE_TYPE + 1)
        elif 'Const' in d: # const node
            assert len(d) == 1 # const node
            const_node = self.const_nodes[d['Const']]
            self.add_double_dir_edge(node.index, const_node.index, AST_EDGE_TYPE, AST_EDGE_TYPE + 1)
        elif 'UNK' in d: # unk node
            assert len(d) == 1 # unk node
            self.add_double_dir_edge(node.index, self.unk_node.index, AST_EDGE_TYPE, AST_EDGE_TYPE + 1)
        else:
            assert type(d) is dict                        
            for key in d:
                if type(d[key]) is dict:
                    child = self.add_node(key)
                    self.add_double_dir_edge(node.index, child.index, AST_EDGE_TYPE, AST_EDGE_TYPE + 1)
                    self.traverse_ast(child, d[key])
                else:
                    # print(key, d[key])
                    assert key == 'cmd' or key == 'OP'
                    child = self.add_node(key + ':' + d[key])
                    self.add_double_dir_edge(node.index, child.index, AST_EDGE_TYPE, AST_EDGE_TYPE + 1)

    def num_nodes(self):
        return len(self.node_list)

    def num_edges(self):
        return len(self.edge_list)

    def add_node(self, node_type, name = None):                
        idx = len(self.node_list)
        node = GraphNode(idx, node_type=node_type, name=name)
        self.node_list.append(node)
        if name is not None:
            key = node_type + '-' + name
            assert key not in self.unique_nodes
            self.unique_nodes[key] = node
        return node

    def add_directed_edge(self, src_idx, dst_idx, edge_type):
        x = self.node_list[src_idx]
        y = self.node_list[dst_idx]
        x.add_out_edge(y, edge_type)
        y.add_in_edge(x, edge_type)
        self.edge_list.append((src_idx, dst_idx, edge_type))

    def add_double_dir_edge(self, src_idx, dst_idx, edge_type_forward, edge_type_backward):
        self.add_directed_edge(src_idx, dst_idx, edge_type_forward)
        self.add_directed_edge(dst_idx, src_idx, edge_type_backward)


def checkallnone(expr_node):
    if expr_node is None:
        return True
    elif expr_node.name in ("&&", "||", ""):
        for childnode in expr_node.children:
            res = checkallnone(childnode)
            if not res:
                return res
        return True
    else:
        return False


if __name__ == '__main__':
    with open(cmd_args.data_root + '/list.txt', 'r') as f:
        for row in f:
            print(row)
            with open(cmd_args.data_root + '/' + row.strip(), 'r') as gf:
                graph_json = json.load(gf)
                g = ProgramGraph(graph_json)

                print(g.num_nodes(), g.num_edges())