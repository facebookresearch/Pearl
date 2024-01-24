### pretty printer
import sys
from .core import CliCmd, add_in_out_args
from .horndb import HornClauseDb, HornRule, load_horn_db_from_file

import z3

def pp_chc_as_rules(db, out):
    fp = None
    if db.has_fixedpoint():
        fp = db.get_fixedpoint()
    else:
        fp = z3.Fixedpoint()
        db.mk_fixedpoint(fp=fp)
    fp.set('print_fixedpoint_extensions', True)
    print('(set-logic ALL)', file=out)
    out.write(fp.sexpr())
    for q in db.get_queries():
        fml = q.mk_query()
        out.write('(query {})\n'.format(fml.sexpr()))

def pp_chc_as_smt(db, out):
    fp = z3.Fixedpoint()
    db.mk_fixedpoint(fp=fp)
    fp.set('print_fixedpoint_extensions', False)
    out.write(fp.sexpr())
    for q in db.get_queries():
        assert(q.has_formula())
        fml = q.get_formula()
        out.write('(assert {})\n'.format(fml.sexpr()))
    out.write('(check-sat)\n')

def pp_chc(db, out, format='rules'):
    if format == 'rules':
        pp_chc_as_rules(db, out)
    else:
        pp_chc_as_smt(db, out)

class ChcPpCmd(CliCmd):
    def __init__(self):
        super().__init__('chcpp', 'Pretty-printer', allow_extra=False)

    def mk_arg_parser(self, ap):
        ap = super().mk_arg_parser(ap)
        ap.add_argument('-o', dest='out_file',
                         metavar='FILE', help='Output file name', default='out.smt2')
        ap.add_argument('in_file',  metavar='FILE', help='Input file')
        ap.add_argument('--format', help='Choice of format', default='rules',
                        choices=['rules', 'chc'])
        return ap

    def run(self, args, extra):
        db = load_horn_db_from_file(args.in_file)
        with open(args.out_file, 'w') as out:
            pp_chc(db, out, format=args.format)

        return 0

if __name__ == '__main__':
    cmd = ChcPpCmd()
    sys.exit(cmd.main(sys.argv[1:]))
