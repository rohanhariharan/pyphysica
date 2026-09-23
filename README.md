# pyphysica

[![PyPI](https://img.shields.io/pypi/v/pyphysica.svg)](https://pypi.org/project/pyphysica/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyphysica.svg)](https://pypi.org/project/pyphysica/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A symbolic + numeric physics DSL for Python. Declare forces as class attributes,
let `pyphysica` integrate them numerically — or derive the equations of motion
symbolically with [SymPy](https://www.sympy.org/).

```python
from physica import Object2, Vector2, continuous, update

ball = Object2("ball", mass=0.45, position=Vector2(0, 10), velocity=Vector2(5, 0))

@continuous(ball)
class Gravity:
    gravity = Vector2(0, -9.81)

@update(ball)
def physics(obj, dt):
    obj.integrate(dt)

for _ in range(100):
    physics(ball, 0.016)

print(ball.position)          # <8.0, -2.5568>
print(ball.kinetic_energy())
```

## Install

```sh
pip install pyphysica
```

From source:

```sh
git clone https://github.com/rohanhariharan/pyphysica
cd pyphysica
pip install -e ".[dev]"
```

## What it does

**Decorator-based forces.** Annotate a class with `@continuous(obj)` to register
each `Vector2` attribute as a per-step force, or `@impulse(obj)` for a one-shot
impulse (`Δv = J/m`).

**Time-varying forces.** `TimeVector` lets a force change over time, either
symbolically in `T` or as a Python callable:

```python
from physica import TimeVector, T

@continuous(obj)
class Engine:
    drive = TimeVector.of(lambda t: Vector2(4 * sp.sin(t), 0))   # callable
    thrust = TimeVector(0, 120.0 - 3 * T)                        # symbolic in T
```

**Symbolic equations of motion.** Since forces may be expressions in `T`,
`Object2` can return them as SymPy:

```python
ax, ay = obj.equation_of_motion()      # expressions in T
ax, ay = obj.acceleration_func()       # T-bound Funcs
vy     = obj.velocity_func()           # integrate once from rest
y      = obj.position_func()           # integrate twice from rest
sol    = obj.closed_form()             # dsolve, zero initial conditions
```

Multi-body systems:

```python
from physica import combine, describe_system

print(describe_system([rocket, booster]))
system = combine([rocket, booster])    # total mass, summed net force, per-body accel
```

**Symbolic math (`Func`/`Lim`).** A thin, ergonomic wrapper over SymPy:

```python
from physica import Func, Lim, x, T
import sympy as sp

f = Func(x**2 - 1)
f(3)            # 8
f[1]            # derivative  -> 2*x
f[-1]           # antiderivative
f[[0, 1]]       # definite integral
(f @ Func(x + 1))     # composition
Lim(sp.sin(x)/x, x, 0)   # 1

g = Func(3 * T + 1, var=T)   # variable-aware: defaults to x
```

**Vectors and more.** `Vector2`, `Vector3`, `Quaternion`, and `Matrix` round out
the math layer.

## Example

See [`examples/example_physics.py`](examples/example_physics.py) for seven
worked examples — projectile with wind and an impulsive kick, a time-varying
drive, multi-object systems, `simulate()` history, `Func` bootstrap over derived
equations, and both the exact and small-angle pendulum.

```sh
python examples/example_physics.py
```

## Tests

```sh
pip install -e ".[dev]"
pytest
```

## Physics notes

- Integration is semi-implicit Euler, which introduces `O(dt)` energy drift.
  Halving the timestep halves the drift.
- `small-angle` checks against the analytic SHM solution agree to `< 1e-3`.
- Equations of motion are derived from Newton's second law: `a = F_net / m`.

## License

MIT
