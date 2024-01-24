import z3

def pushed_solver (solver):
  """
  Context Manager helper to push a solver.

  s = Solver ()
  s.add (c1)
  # at this point s is not yet pushed
  with pushed_solver (s) as ps:
    # here ps is s after a push
    ps.add (c2)
    ps.add (c3)
    print ps
  # here s is just like it was before the 'with' statement
  print s
  """
  class PushSolverContextManager (object):
    def __init__ (self, solver):
      self.solver = solver

    def __enter__(self):
      self.solver.push ()
      return self.solver

    def __exit__ (self, exc_type, exc_value, traceback):
      self.solver.pop ()
      # do not suppress any exception
      return False

  return PushSolverContextManager (solver)

