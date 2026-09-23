import numpy as np
import sympy as sp
from sympy import *
from sympy import solve
import inspect
import random
import math
import copy
from functools import wraps


import numpy as np
import sympy as sp
from sympy import solve, sympify
import math
 
# Canonical free variable
x = sp.Symbol("x")

# Canonical time symbol (used by TimeVector and symbolic equations of motion)
T = sp.Symbol("T")
 
sp.Symbol.__rshift__ = lambda self, other: (self, other) if isinstance(other, (int, float)) else TypeError("Right-shift only supports numeric literals, not {}".format(type(other)))
 
def sign(integer):
    return "" if integer < 0 else "+"
 
 
# ---------------------------------------------------------------------------
# Quaternion
# ---------------------------------------------------------------------------
 
class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = sp.sympify(w)
        self.x = sp.sympify(x)
        self.y = sp.sympify(y)
        self.z = sp.sympify(z)
 
    def __repr__(self):
        # Use evalf so symbolic components render as floats when possible,
        # but fall back to the symbolic form cleanly.
        def fmt(v):
            try:
                return f"{float(v):.4g}"
            except (TypeError, ValueError):
                return str(v)
        return f"Quaternion({fmt(self.w)}, {fmt(self.x)}, {fmt(self.y)}, {fmt(self.z)})"
 
    def norm(self):
        return sp.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
 
    def normalize(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize a zero quaternion.")
        self.w /= n
        self.x /= n
        self.y /= n
        self.z /= n
        return self
 
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)
 
    def inverse(self):
        n2 = self.norm() ** 2
        if n2 == 0:
            raise ZeroDivisionError("Cannot invert a zero quaternion.")
        return self.conjugate() * (sp.Integer(1) / n2)
 
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
            x_ = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
            y_ = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
            z_ = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
            return Quaternion(w, x_, y_, z_)
        return Quaternion(self.w*other, self.x*other, self.y*other, self.z*other)
 
    def rotate_vector(self, vector):
        qv = Quaternion(0, vector.x, vector.y, vector.z)
        qr = self * qv * self.inverse()
        return Vector3(qr.x, qr.y, qr.z)
 
 
# ---------------------------------------------------------------------------
# Vector3
# ---------------------------------------------------------------------------
 
class Vector3:
    __slots__ = ("x", "y", "z")
 
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = sp.sympify(x)
        self.y = sp.sympify(y)
        self.z = sp.sympify(z)
 
    def _fmt(self, v):
        try:
            return f"{float(v):.6g}"
        except (TypeError, ValueError):
            return str(v)
 
    def __repr__(self):
        return f"Vector3({self._fmt(self.x)}, {self._fmt(self.y)}, {self._fmt(self.z)})"
 
    def __str__(self):
        return f"<{self.x}, {self.y}, {self.z}>"
 
    @property
    def components(self):
        return f"{self.x}i + {self.y}j + {self.z}k"
 
    def copy(self):
        return Vector3(self.x, self.y, self.z)
 
    def as_tuple(self):
        return (self.x, self.y, self.z)
 
    def as_array(self):
        return np.array([float(self.x), float(self.y), float(self.z)])
 
    def norm(self):
        return sp.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
 
    def norm_sq(self):
        return self.x*self.x + self.y*self.y + self.z*self.z
 
    def __abs__(self):
        return self.norm()
 
    def normalize(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        self.x /= n
        self.y /= n
        self.z /= n
        return self
 
    def normalized(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return Vector3(self.x/n, self.y/n, self.z/n)
 
    def __add__(self, other):
        return Vector3(self.x+other.x, self.y+other.y, self.z+other.z)
 
    def __sub__(self, other):
        return Vector3(self.x-other.x, self.y-other.y, self.z-other.z)
 
    def __mul__(self, scalar):
        return Vector3(self.x*scalar, self.y*scalar, self.z*scalar)
 
    def __rmul__(self, scalar):
        return self * scalar
 
    def __truediv__(self, scalar):
        return Vector3(self.x/scalar, self.y/scalar, self.z/scalar)
 
    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)
 
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
 
    def cross(self, other):
        return Vector3(
            self.y*other.z - self.z*other.y,
            self.z*other.x - self.x*other.z,
            self.x*other.y - self.y*other.x
        )
 
    def distance_to(self, other):
        return (self - other).norm()
 
    def project(self, other):
        d = other.norm_sq()
        if d == 0:
            raise ZeroDivisionError("Projection onto zero vector")
        return other * (self.dot(other) / d)
 
    def reject(self, other):
        return self - self.project(other)
 
    def angle_to(self, other):
        m = float(self.norm()) * float(other.norm())
        if m == 0:
            raise ZeroDivisionError("Angle undefined for zero vector")
        return math.acos(max(-1.0, min(1.0, float(self.dot(other)) / m)))
 
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z
 
    def __getitem__(self, i):
        if i == 0: return self.x
        if i == 1: return self.y
        if i == 2: return self.z
        raise IndexError
 
    def __eq__(self, other):
        return isinstance(other, Vector3) and \
               self.x == other.x and self.y == other.y and self.z == other.z
 
    def rotated(self, quaternion):
        return quaternion.rotate_vector(self)
 
 
# ---------------------------------------------------------------------------
# Vector2
# ---------------------------------------------------------------------------
 
class Vector2:
    __slots__ = ("x", "y", "type")
 
    def __init__(self, x=0.0, y=0.0):
        self.x = sp.sympify(x)
        self.y = sp.sympify(y)
        self.type = ""
 
    def _fmt(self, v):
        try:
            return f"{float(v):.6g}"
        except (TypeError, ValueError):
            return str(v)
 
    def __repr__(self):
        return f"Vector2({self._fmt(self.x)}, {self._fmt(self.y)})"
 
    def __str__(self):
        return f"<{self.x}, {self.y}>"
 
    def copy(self):
        return Vector2(self.x, self.y)
 
    def as_tuple(self):
        return (self.x, self.y)
 
    def as_array(self):
        return np.array([float(self.x), float(self.y)])
 
    def norm(self):
        return sp.sqrt(self.x*self.x + self.y*self.y)
 
    def __abs__(self):
        return self.norm()
 
    def norm_sq(self):
        return self.x*self.x + self.y*self.y
 
    def normalize(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        self.x /= n
        self.y /= n
        return self
 
    def normalized(self):
        n = self.norm()
        if n == 0:
            raise ZeroDivisionError("Cannot normalize zero vector")
        return Vector2(self.x/n, self.y/n)
 
    def __add__(self, other):
        return Vector2(self.x+other.x, self.y+other.y)
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            if isinstance(other, Vector2):
                return other.__add__(self)
            else:
                raise TypeError(f"Unsupported operand for types {type(other)} and Vector2.")
 
    def __sub__(self, other):
        return Vector2(self.x-other.x, self.y-other.y)
 
    def __mul__(self, scalar):
        return Vector2(self.x*scalar, self.y*scalar)
 
    def __rmul__(self, scalar):
        return self * scalar
 
    def __truediv__(self, scalar):
        return Vector2(self.x/scalar, self.y/scalar)
 
    def __neg__(self):
        return Vector2(-self.x, -self.y)
 
    def dot(self, other):
        return self.x*other.x + self.y*other.y
 
    def cross(self, other):
        return self.x*other.y - self.y*other.x
 
    def angle(self):
        return sp.atan2(self.y, self.x)
 
    def angle_to(self, other):
        if not isinstance(other, Vector2):
            raise TypeError(f"angle_to requires Vector2, not {type(other)}")
        m = self.norm() * other.norm()
        if m == 0:
            raise ZeroDivisionError("Angle undefined for zero vector")
        return sp.acos(sp.Rational(max(-1, min(1, self.dot(other)/m))))
 
    def distance_to(self, other):
        return (self - other).norm()
 
    def project(self, other):
        d = other.norm_sq()
        if d == 0:
            raise ZeroDivisionError("Projection onto zero vector")
        return other * (self.dot(other) / d)
 
    def reject(self, other):
        return self - self.project(other)
 
    def rotate(self, theta):
        c = sp.cos(theta)
        s = sp.sin(theta)
        return Vector2(self.x*c - self.y*s, self.x*s + self.y*c)
 
    def perpendicular(self):
        return Vector2(-self.y, self.x)
 
    def lerp(self, other, t):
        return Vector2(self.x + (other.x-self.x)*t, self.y + (other.y-self.y)*t)
 
    @staticmethod
    def from_polar(r, theta):
        return Vector2(r*sp.cos(theta), r*sp.sin(theta))
 
    def __iter__(self):
        yield self.x
        yield self.y
 
    def __getitem__(self, i):
        if i == 0: return self.x
        if i == 1: return self.y
        raise IndexError
 
    def __eq__(self, other):
        return isinstance(other, Vector2) and self.x == other.x and self.y == other.y
 
 
# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------
 
class Matrix:
    def __init__(self, *rows):
        # Accept either Matrix(*rows) or Matrix(list_of_rows)
        if len(rows) == 1 and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
            rows = rows[0]
 
        if not rows:
            raise ValueError("Matrix cannot be empty")
        if not all(len(row) == len(rows[0]) for row in rows):
            raise ValueError("Rows must have equal length")
 
        self.data = [list(row) for row in rows]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
 
    def __str__(self):
        return "\n".join(" ".join(str(v) for v in row) for row in self.data)
 
    def __repr__(self):
        return f"Matrix({self.data})"
 
    def shape(self):
        return (self.rows, self.cols)
 
    def copy(self):
        return Matrix([row[:] for row in self.data])
 
    def __getitem__(self, idx):
        return self.data[idx]
 
    def __eq__(self, other):
        return isinstance(other, Matrix) and self.data == other.data
 
    def __add__(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrix dimensions must match for addition")
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def __sub__(self, other):
        if self.shape() != other.shape():
            raise ValueError("Matrix dimensions must match for subtraction")
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def __mul__(self, scalar):
        return Matrix([[scalar * v for v in row] for row in self.data])
 
    __rmul__ = __mul__
 
    def __matmul__(self, other):
        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(f"Cannot multiply ({self.rows}x{self.cols}) @ ({other.rows}x{other.cols})")
            return Matrix([
                [sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                 for j in range(other.cols)]
                for i in range(self.rows)
            ])
 
    def __pow__(self, n):
        if self.rows != self.cols:
            raise ValueError("Matrix must be square for exponentiation")
        result = Matrix.identity(self.rows)
        base = self.copy()
        n = int(n)
        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n //= 2
        return result
 
    @property
    def T(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])
 
    def trace(self):
        if self.rows != self.cols:
            raise ValueError("Trace requires a square matrix")
        return sum(self.data[i][i] for i in range(self.rows))
 
    def det(self):
        if self.rows != self.cols:
            raise ValueError("Determinant requires a square matrix")
        if self.rows == 1:
            return self.data[0][0]
        if self.rows == 2:
            a, b = self.data[0]
            c, d = self.data[1]
            return a*d - b*c
        total = 0
        for col in range(self.cols):
            sub = [row[:col] + row[col+1:] for row in self.data[1:]]
            total += ((-1)**col) * self.data[0][col] * Matrix(sub).det()
        return total
 
    def minor(self, r, c):
        sub = [row[:c] + row[c+1:] for i, row in enumerate(self.data) if i != r]
        return Matrix(sub).det()
 
    def cofactors(self):
        return Matrix([
            [((-1)**(i+j)) * self.minor(i, j) for j in range(self.cols)]
            for i in range(self.rows)
        ])
 
    def adj(self):
        return self.cofactors().T
 
    def inv(self):
        d = self.det()
        if d == 0:
            raise ValueError("Matrix is singular (not invertible)")
        return (1/d) * self.adj()
 
    def rank(self):
        m = [row[:] for row in self.data]
        rank = 0
        for c in range(self.cols):
            pivot = next((r for r in range(rank, self.rows) if m[r][c] != 0), None)
            if pivot is None:
                continue
            m[rank], m[pivot] = m[pivot], m[rank]
            pv = m[rank][c]
            m[rank] = [v / pv for v in m[rank]]
            for r in range(self.rows):
                if r != rank:
                    f = m[r][c]
                    m[r] = [iv - f*rv for rv, iv in zip(m[rank], m[r])]
            rank += 1
        return rank
 
    def rref(self):
        m = [row[:] for row in self.data]
        lead = 0
        for r in range(self.rows):
            if lead >= self.cols:
                break
            i = r
            while m[i][lead] == 0:
                i += 1
                if i == self.rows:
                    i = r
                    lead += 1
                    if lead == self.cols:
                        return Matrix(m)
            m[i], m[r] = m[r], m[i]
            lv = m[r][lead]
            m[r] = [val/lv for val in m[r]]
            for i in range(self.rows):
                if i != r:
                    lv = m[i][lead]
                    m[i] = [iv - lv*rv for rv, iv in zip(m[r], m[i])]
            lead += 1
        return Matrix(m)
 
    def solve(self, b):
        if isinstance(b, Matrix):
            b_data = [row[0] if len(row) == 1 else row for row in b.data]
        else:
            b_data = b
        aug = [self.data[i] + [b_data[i]] for i in range(self.rows)]
        rref = Matrix(aug).rref().data
        return [row[-1] for row in rref]
 
    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
 
    @staticmethod
    def zeros(r, c):
        return Matrix([[0]*c for _ in range(r)])
 
 
# ---------------------------------------------------------------------------
# Lim  — fixed signature: Lim(expr, var, point)
# ---------------------------------------------------------------------------
 
class Lim:
    """
    Compute symbolic limits.
 
    Usage:
        Lim(sin(x)/x, x, 0)          -> 1
        Lim(Func(sin(x)/x), x, 0)    -> 1
    """
    def __init__(self, expr, var=None, point=None):
        # Resolve Func wrappers
        if isinstance(expr, Func):
            expr = expr.expr
 
        self.expr = expr
        self.var = var if var is not None else x
        self.point = point
 
    def __rshift__(self, target):
        """Support Lim(expr, x) >> 0  syntax."""
        return Lim(self.expr, self.var, target)
 
    def evaluate(self):
        if self.point is None:
            raise ValueError("Limit point not set.")
        return sp.limit(self.expr, self.var, self.point)
    
    def __call__(self, expr):
        self.expr = expr
        return self.evaluate()

    def __eq__(self, other):
        if self.point is None:
            return NotImplemented
        return sp.simplify(self.evaluate() - sp.sympify(other)) == 0
 
    def __repr__(self):
        if self.point is not None:
            return str(self.evaluate())
        return f"Lim({self.expr}, {self.var}, ?)"
 
    def __str__(self):
        return self.__repr__()
 
 
# ---------------------------------------------------------------------------
# Func
# ---------------------------------------------------------------------------
 
class Func:
    """
    A symbolic function wrapper using SymPy.
 
    f = Func(x**2 - 1)
    f(3)            -> 8
    f[1]            -> first derivative as Func
    f[-1]           -> indefinite integral as Func
    f[[0, 1]]       -> definite integral from 0 to 1
    f @ g           -> composition f(g(x))
    f.plot()        -> plot (handled by backend)
    f.derivative(n) -> nth derivative
    f.integrate(n)  -> nth antiderivative
    f.series(n)     -> Taylor series to order n
    f.inverse()     -> symbolic inverse (first branch)
    f >> sym        -> solve f(x) = 0 for sym
    """
    def __init__(self, expression, var=None):
        if callable(expression) and not isinstance(expression, sp.Basic):
            raise ValueError("Use symbolic SymPy expressions only, not callables.")
        self.expr = sp.sympify(expression)
        self.var = var if var is not None else x
        self._vars = sorted(self.expr.free_symbols, key=lambda s: s.name)

    def _lambdify(self):
        """Build a numeric evaluator — prefers numpy for robustness."""
        return sp.lambdify([self.var], self.expr, modules=["numpy", "sympy"])

    def __call__(self, val):
        return self._lambdify()(val)

    def inverse(self):
        y = sp.Symbol('_y')
        sol = sp.solve(self.expr - y, self.var)
        if not sol:
            raise ValueError("Inverse not found or not unique.")
        return Func(sol[0].subs(y, self.var), var=self.var)

    def solve(self, sym=None):
        sym = sym or self.var
        return sp.solve(self.expr, sym)
 
    def __rshift__(self, sym):
        if isinstance(sym, sp.Symbol):
            return sp.solve(self.expr, sym)
        raise TypeError("Right-shift argument must be a sympy.Symbol")
 
    def __add__(self, other):
        return Func(self.expr + (other.expr if isinstance(other, Func) else other), var=self.var)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return Func(self.expr - (other.expr if isinstance(other, Func) else other), var=self.var)

    def __rsub__(self, other):
        return Func(other - self.expr, var=self.var)

    def __mul__(self, other):
        return Func(self.expr * (other.expr if isinstance(other, Func) else other), var=self.var)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return Func(self.expr / (other.expr if isinstance(other, Func) else other), var=self.var)

    def __rtruediv__(self, other):
        return Func(other / self.expr, var=self.var)
 
    def __matmul__(self, other):
        if not isinstance(other, Func):
            raise TypeError(f"Cannot compose Func and {type(other)}")
        return Func(self.expr.subs(self.var, other.expr), var=self.var)

    def __pow__(self, n):
        if not isinstance(n, int):
            raise TypeError("Only integer powers are supported.")
        if n == 0:
            return Func(sp.S.One, var=self.var)
        base = self if n > 0 else self.inverse()
        result = base
        for _ in range(abs(n) - 1):
            result = result @ base
        return result
 
    def __eq__(self, other):
        if isinstance(other, Func):
            return sp.simplify(self.expr - other.expr) == 0
        return NotImplemented
 
    def __str__(self):
        return str(self.expr)
 
    def __repr__(self):
        return str(self.expr)
 
    def simplify(self):
        return Func(sp.simplify(self.expr), var=self.var)

    def derivative(self, n=1):
        return Func(sp.diff(self.expr, self.var, n), var=self.var)

    def integrate(self, n=1):
        result = self
        for _ in range(n):
            result = Func(sp.integrate(result.expr, self.var), var=self.var)
        return result
 
    def partial(self, var, n=1):
        if not isinstance(var, sp.Symbol):
            raise TypeError("Variable must be a sympy.Symbol")
        return Func(sp.diff(self.expr, var, n))
 
    def gradient(self, variables=None):
        variables = variables or self._vars
        return [self.partial(v) for v in variables]
 
    def hessian(self, *variables):
        variables = variables or self._vars
        n = len(variables)
        return [[self.partial(variables[i]).partial(variables[j]) for j in range(n)] for i in range(n)]
 
    def series(self, n=6):
        return sp.series(self.expr, self.var, n=n)

    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= 0:
                return Func(sp.diff(self.expr, self.var, key), var=self.var)
            # Negative: nth antiderivative
            result = self
            for _ in range(abs(key)):
                result = Func(sp.integrate(result.expr, self.var), var=self.var)
            return result
        if isinstance(key, (list, tuple)) and len(key) == 2:
            lower, upper = key
            return sp.integrate(self.expr, (self.var, lower, upper))
        raise TypeError("Key must be an int (derivative order) or [lower, upper] (definite integral).")


class TimeVector:
    """
    A Vector2 whose components may vary with time.

    Two construction forms:
        TimeVector(x, y)                       # symbolic components in T
        TimeVector.of(lambda t: Vector2(...))  # callable, evaluated at time t

    Examples:
        TimeVector(0, 120.0 - 3 * T)
        TimeVector.of(lambda t: Vector2(4 * sp.sin(t), 0))

    Use .at(t) for a numeric Vector2 at time t, and .symbolic() for the
    expression in T (used by the symbolic equations of motion).
    """

    def __init__(self, x=0.0, y=0.0, fn=None):
        self.fn = fn
        self.type = ""
        if fn is None:
            self._x = sp.sympify(x)
            self._y = sp.sympify(y)
        else:
            self._x = None
            self._y = None

    @classmethod
    def of(cls, fn):
        if not callable(fn):
            raise TypeError("TimeVector.of expects a callable")
        return cls(fn=fn)

    @property
    def x(self):
        """Symbolic x-component (expressions in T)."""
        return self.symbolic().x

    @property
    def y(self):
        """Symbolic y-component (expressions in T)."""
        return self.symbolic().y

    @staticmethod
    def _as_vector(value, where):
        if isinstance(value, Vector2):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return Vector2(value[0], value[1])
        raise TypeError(f"TimeVector callable must return Vector2 (or a 2-tuple), got {type(value)} from {where}")

    def symbolic(self):
        """Return a Vector2 of expressions in T."""
        if self.fn is not None:
            try:
                return self._as_vector(self.fn(T), "T")
            except Exception:
                return self._as_vector(self.fn(0), "t=0")
        return Vector2(self._x, self._y)

    def at(self, t):
        """Return the numeric Vector2 at time t."""
        if self.fn is not None:
            return self._as_vector(self.fn(t), f"t={t}")
        return Vector2(self._x.subs(T, t), self._y.subs(T, t))

    def norm(self):
        return self.symbolic().norm()

    def copy(self):
        if self.fn is not None:
            return TimeVector(fn=self.fn)
        return TimeVector(self._x, self._y)

    def __repr__(self):
        if self.fn is not None:
            return f"TimeVector.of({self.fn})"
        return f"TimeVector({self._x}, {self._y})"

    def __str__(self):
        return str(self.symbolic())


class Time:
    def __init__(self):
        self.t = 0

    def up(self, val:int | float = 0):
        self.t += val

    def down(self, val:int | float = 0):
            self.t -= val

    def reset(self):
            self.t = 0

class Object2:
    # ------------------ Initialization & Representation ------------------
    def __init__(self, name="Object", mass=1.0, position:Vector2=None, velocity:Vector2=None, acceleration:Vector2=None, radius=1.0):
        self.name = name
        self.mass = mass
        self.position = position if position is not None else Vector2()
        self.velocity = velocity if velocity is not None else Vector2()
        self.acceleration = acceleration if acceleration is not None else Vector2()
        self.radius = radius
        self.angle = 0.0
        self.angular_velocity = 0.0
        self.torque = 0.0
        self.moment = 0.5 * mass * radius ** 2
        self.time = 0.0
        self.forces: list = []
    
    def __repr__(self):
        return f"Object2(name={self.name}, mass={self.mass}, pos={self.position}, vel={self.velocity})"
    
    def __str__(self):
        return f"{self.name}: pos={self.position}, vel={self.velocity}, mass={self.mass}"
    
    # ------------------ Force Management ------------------
    def apply_force_continuous(self, force):
        """Add a continuous force. Accepts Vector2 or TimeVector."""
        if not isinstance(force, (Vector2, TimeVector)):
            raise TypeError(f"Force must be Vector2 or TimeVector, got {type(force)}")
        force.type = "continuous"
        if not any(force is existing for existing in self.forces):
            self.forces.append(force)
        self.sortforces()
        return f"Applied continuous force of magnitude {self._vector_at(force, self.time).norm()}"

    def apply_force_impulse(self, force):
        """Add an instantaneous impulse force. Accepts Vector2 or TimeVector."""
        if not isinstance(force, (Vector2, TimeVector)):
            raise TypeError(f"Force must be Vector2 or TimeVector, got {type(force)}")
        force.type = "impulse"
        if not any(force is existing for existing in self.forces):
            self.forces.append(force)
        self.sortforces()
        return self.apply_impulse(self._vector_at(force, self.time))

    @staticmethod
    def _vector_at(force, t):
        """Resolve a force to a Vector2 at time t (evaluating TimeVector)."""
        if isinstance(force, TimeVector):
            return force.at(t)
        return force

    @property
    def continuous_forces(self):
        return [force for force in self.forces if force.type == "continuous"]

    @property
    def impulse_forces(self):
        return [force for force in self.forces if force.type == "impulse"]

    @property
    def net_force(self):
        """Sum of all continuous forces, evaluated at the current time."""
        out = Vector2()
        for force in self.continuous_forces:
            out = out + self._vector_at(force, self.time)
        return out

    def net_force_at(self, t):
        """Sum of continuous forces evaluated at an arbitrary time t."""
        out = Vector2()
        for force in self.continuous_forces:
            out = out + self._vector_at(force, t)
        return out

    def clear_forces(self):
        """Clear all forces"""
        self.forces.clear()

    def clear_impulses(self):
        """Remove one-shot impulse forces without touching continuous ones"""
        self.forces = [force for force in self.forces if force.type != "impulse"]

    def sortforces(self) -> None:
        """Kept for backwards compatibility; force lists are now computed properties."""
        return None
    
    # ------------------ Motion / Physics ------------------
    def update(self, dt):
        return self.integrate(dt)

    def integrate(self, dt):
        self.acceleration = self.net_force / self.mass
        self.position = self.position + self.velocity*dt + self.acceleration*(0.5*dt*dt)
        self.velocity = self.velocity + self.acceleration*dt
        self.angular_velocity = self.angular_velocity + (self.torque / self.moment)*dt
        self.angle = self.angle + self.angular_velocity*dt
        self.time = self.time + dt
        return self

    # ------------------ Symbolic (Func) Integration ------------------
    def net_force_symbolic(self):
        """Return the net continuous force as a Vector2 of expressions in T."""
        out = Vector2()
        for force in self.continuous_forces:
            if isinstance(force, TimeVector):
                out = out + force.symbolic()
            else:
                out = out + force
        return out

    def equation_of_motion(self):
        """Return (ax(T), ay(T)) SymPy expressions from Newton's second law."""
        net = self.net_force_symbolic()
        return (sp.simplify(net.x / self.mass), sp.simplify(net.y / self.mass))

    def acceleration_func(self):
        """Return (ax, ay) as T-bound Funcs."""
        ax, ay = self.equation_of_motion()
        return Func(ax, var=T), Func(ay, var=T)

    def velocity_func(self):
        """Return (vx, vy) as T-bound Funcs, integrating acceleration from rest."""
        ax, ay = self.acceleration_func()
        return ax[-1], ay[-1]

    def position_func(self):
        """Return (x, y) as T-bound Funcs, double-integrating acceleration from rest."""
        ax, ay = self.acceleration_func()
        return ax[-2], ay[-2]

    def closed_form(self):
        """Solve the equations of motion symbolically with dsolve (zero ICs)."""
        ax, ay = self.equation_of_motion()
        solutions = []
        for component in (ax, ay):
            f = sp.Function("_q")(T)

            # dsolve can recurse on Float RHS in some SymPy/Python builds; rationalize.
            rhs = component
            if rhs.has(sp.Float):
                rhs = sp.nsimplify(rhs, rational=True)
            try:
                sol = sp.dsolve(sp.Eq(f.diff(T, 2), rhs), f)
                solutions.append(sol.rhs)
            except Exception:
                solutions.append(sp.integrate(sp.integrate(rhs, T), T))
        return tuple(solutions)

    def describe(self):
        """Human-readable summary of the symbolic system."""
        net = self.net_force_symbolic()
        ax, ay = self.equation_of_motion()
        lines = [
            f"{self.name} (mass={self.mass})",
            f"  net force F(T) = {net}",
            f"  acceleration a(T) = ({ax}, {ay})",
        ]
        return "\n".join(lines)

    def apply_gravity(self, gravity):
        self.apply_force_continuous(Vector2(0, gravity))
    
    def apply_friction(self, normal_force, coefficient):
        """Apply kinetic friction opposite to velocity"""
        speed = self.speed()
        if speed == 0:
            return Vector2()
        magnitude = abs(coefficient * normal_force)
        friction = self.velocity.normalized() * -magnitude
        self.apply_force_continuous(friction)
        return friction

    def apply_drag(self, fluid_density, drag_coeff, area):
        """Apply quadratic drag: F = -1/2 rho C A |v| v"""
        speed = self.speed()
        if speed == 0:
            return Vector2()
        drag = self.velocity.normalized() * (-0.5 * fluid_density * drag_coeff * area * speed ** 2)
        self.apply_force_continuous(drag)
        return drag

    def move_to(self, position):
        """Directly set position"""
        self.position = position

    def set_velocity(self, velocity):
        """Directly set velocity"""
        self.velocity = velocity

    # ------------------ State / Info ------------------
    def state(self):
        """Return dictionary with position, velocity, mass, radius, name"""
        return {
            "name": self.name,
            "mass": self.mass,
            "radius": self.radius,
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "acceleration": self.acceleration.copy(),
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
        }

    def kinetic_energy(self):
        """Return kinetic energy: 1/2 m v^2"""
        return 0.5 * self.mass * self.speed() ** 2

    def momentum(self):
        """Return momentum: m * v"""
        return self.velocity * self.mass

    def speed(self):
        """Return magnitude of velocity"""
        return self.velocity.norm()

    def distance_to(self, other):
        """Return Euclidean distance to another Object2"""
        return self.position.distance_to(other.position)

    def direction_to(self, other):
        """Return normalized Vector2D pointing to another Object2"""
        return (other.position - self.position).normalized()

    # ------------------ Collision & Interaction ------------------
    def check_collision(self, other):
        """Return True if colliding with another Object2"""
        return self.distance_to(other) <= self.radius + other.radius

    def resolve_collision(self, other, restitution=0.9):
        """Basic elastic collision response between two equal-radius bodies"""
        if not self.check_collision(other):
            return False
        normal = (other.position - self.position)
        if normal.norm() == 0:
            normal = Vector2(1, 0)
        normal = normal.normalized()
        relative = (other.velocity - self.velocity).dot(normal)
        if relative > 0:
            return False
        impulse = -(1 + restitution) * relative / (1 / self.mass + 1 / other.mass)
        self.velocity = self.velocity - normal * (impulse / self.mass)
        other.velocity = other.velocity + normal * (impulse / other.mass)
        overlap = self.radius + other.radius - self.distance_to(other)
        if overlap > 0:
            correction = normal * (overlap / 2)
            self.position = self.position - correction
            other.position = other.position + correction
        return True

    def apply_spring(self, other, k, rest_length):
        """Apply spring force to other object using Hooke's Law: F = -k x"""
        offset = other.position - self.position
        length = offset.norm()
        if length == 0:
            return other
        extension = length - rest_length
        direction = offset.normalized()
        other.apply_force_continuous(direction * (k * extension))
        self.apply_force_continuous(direction * (-k * extension))
        return other

    def apply_torque(self, torque):
        """Apply rotational torque"""
        self.torque = torque
        return self

    # ------------------ Dunder Methods for Arithmetic / Comparison ------------------
    def __add__(self, other):
        result = self.clone()
        if isinstance(other, Object2):
            result.position = self.position + other.position
            result.velocity = self.velocity + other.velocity
        else:
            result.position = self.position + other
            result.velocity = self.velocity + other
        return result

    def __sub__(self, other):
        result = self.clone()
        if isinstance(other, Object2):
            result.position = self.position - other.position
            result.velocity = self.velocity - other.velocity
        else:
            result.position = self.position - other
            result.velocity = self.velocity - other
        return result

    def __mul__(self, scalar):
        result = self.clone()
        result.position = self.position * scalar
        result.velocity = self.velocity * scalar
        result.mass = self.mass * scalar
        return result

    def __truediv__(self, scalar):
        result = self.clone()
        result.position = self.position / scalar
        result.velocity = self.velocity / scalar
        result.mass = self.mass / scalar
        return result

    def __eq__(self, other):
        if not isinstance(other, Object2):
            return NotImplemented
        return self.name == other.name and self.mass == other.mass and self.position == other.position

    def __lt__(self, other):
        return self.mass < other.mass

    def __le__(self, other):
        return self.mass <= other.mass

    def __gt__(self, other):
        return self.mass > other.mass

    def __ge__(self, other):
        return self.mass >= other.mass

    def __hash__(self):
        return id(self)

    def clone(self):
        """Return a deep copy of this object"""
        return copy.deepcopy(self)

    def zero_velocity(self):
        """Stop the object"""
        self.velocity = Vector2()
        self.angular_velocity = 0.0
        return self

    def zero_forces(self):
        """Remove all forces without updating"""
        self.clear_forces()
        return self

    def apply_impulse(self, impulse):
        """Instant velocity change: Δv = J/m"""
        self.velocity = self.velocity + impulse / self.mass
        return self.velocity


def _vectors_in(_class):
    """Yield the Vector2 attributes declared on a decorated class."""
    for _, value in vars(_class).items():
        if isinstance(value, (Vector2, TimeVector)):
            yield value


def _register(target_object, force, kind: str):
    """Attach a copy of a declared force to a target with the given kind."""
    force = force.copy()
    force.type = kind
    if kind == "impulse":
        target_object.apply_force_impulse(force)
    else:
        target_object.apply_force_continuous(force)
    return force


def continuous(target_object: Object2):
    """@continuous(obj) on a class: apply each Vector2 attribute as a
    continuous (every-step) force on obj. On a function, run it each step."""
    def decorator(_class):
        if inspect.isfunction(_class):
            _class.force_type = "continuous"
            return _class
        applied = [_register(target_object, value, "continuous") for value in _vectors_in(_class)]
        _class.forces = applied
        return _class
    return decorator


def impulse(target_object: Object2):
    """@impulse(obj) on a class: apply each Vector2 attribute as a one-shot
    impulse force on obj (Δv = J/m)."""
    def decorator(_class):
        if inspect.isfunction(_class):
            _class.force_type = "impulse"
            return _class
        applied = [_register(target_object, value, "impulse") for value in _vectors_in(_class)]
        _class.forces = applied
        return _class
    return decorator


def update(target_object: Object2, dt: int = 1, length=100):
    """Two forms:

    @update(obj)                       # on a function: a per-tick system
    def physics(o, dt): ...

    @update(obj, dt=0.016, length=100) # on a class holding a Time():
    class Newtonian:
        time = Time()
    The class form advances `time` and `obj` for `length` steps and stores the
    per-step `obj.state()` in `Newtonian.history`.
    """
    def decorator(_class):
        if inspect.isfunction(_class):
            history = []

            @wraps(_class)
            def system(obj=target_object, step=dt, *args, **kwargs):
                result = _class(obj, step, *args, **kwargs)
                if isinstance(obj, Object2):
                    history.append((obj.state(), step))
                return result

            system.history = history
            system.target = target_object
            return system

        time = next((value for value in vars(_class).values() if isinstance(value, Time)), None)
        if time is None:
            raise ValueError("Update rule requires a Time object")
        history = {}
        for _ in range(length):
            time.up(dt)
            target_object.integrate(dt)
            history[time.t] = target_object.state()
        _class.time = time
        _class.history = history
        return _class
    return decorator


def simulate(objects, dt=1, length=100, time: Time = None):
    """Run every object's integrate(dt) for `length` steps.

    Continuous forces are re-evaluated each step; impulses are cleared once
    consumed. Returns {t: [state, ...]} so the run can be replayed.
    """
    if isinstance(objects, Object2):
        objects = [objects]
    if time is None:
        time = Time()
    history = {}
    for _ in range(length):
        time.up(dt)
        states = []
        for obj in objects:
            obj.integrate(dt)
            states.append(obj.state())
        history[time.t] = states
        for obj in objects:
            obj.clear_impulses()
    return history


def combine(objects):
    """Combine the symbolic equations of motion of several objects.

    Returns one System dict with the per-object acceleration components, the
    total mass, and the summed net force, all as expressions in T.
    """
    if isinstance(objects, Object2):
        objects = [objects]
    objects = list(objects)
    if not objects:
        raise ValueError("combine() requires at least one object")

    net = Vector2()
    for obj in objects:
        net = net + obj.net_force_symbolic()

    return {
        "objects": objects,
        "mass": sum(obj.mass for obj in objects),
        "net_force": net,
        "accelerations": [obj.equation_of_motion() for obj in objects],
    }


def describe_system(objects):
    """Human-readable multi-object system summary (forces + accelerations in T)."""
    if isinstance(objects, Object2):
        objects = [objects]
    objects = list(objects)
    combined = combine(objects)
    lines = [f"System of {len(objects)} object(s), total mass = {combined['mass']}"]
    for obj in objects:
        ax, ay = obj.equation_of_motion()
        lines.append(f"  {obj.name}: a(T) = ({ax}, {ay})")
    lines.append(f"  total net force F(T) = {combined['net_force']}")
    return "\n".join(lines)







