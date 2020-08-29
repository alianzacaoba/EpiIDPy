import sympy
from sympy.parsing.sym_expr import SymPyExpression

state = sympy.symbols('s e i')
print(state)
src = '''
    -ne
'''

p = SymPyExpression(src, 'f')
p.convert_to_python()
