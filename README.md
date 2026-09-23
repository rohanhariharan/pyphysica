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

## Integrators

Each `Object2` carries an `integrator`, set at construction or per call:

```python
obj = Object2("ball", mass=1.0, integrator="verlet")   # default
obj.integrate(dt)                 # uses obj.integrator
obj.integrate(dt, "rk4")          # override for this call
simulate(obj, dt=0.01, length=100, integrator="euler-cromer")
```

Built-in schemes (`INTEGRATORS`), all advancing `integration` in place:

| name | order | symplectic | notes |
|---|---|---|---|
| `verlet` (default) | 2 | yes | Velocity Verlet. Bounded energy, no secular drift. Exact for constant force. |
| `euler-cromer` | 1 | yes | Semi-implicit Euler. Bounded energy, but first-order. |
| `rk4` | 4 | no | Runge-Kutta 4. Highest accuracy for smooth/time-varying forces; energy may drift on long conservative runs. |

Custom schemes can be registered by adding a `(obj, dt) -> obj` callable to
`INTEGRATORS`, or passed directly to `integrate`.

## Physics notes

- The default is Velocity Verlet: 2nd-order symplectic, so energy stays bounded
  on conservative systems (measured `2e-4` on a spring at `dt=0.01`).
- `small-angle` checks against the analytic SHM solution agree to `< 1e-3`.
- Equations of motion are derived from Newton's second law: `a = F_net / m`.
- The angular DOF uses an explicit update (the torque is taken as constant over
  a step) for all schemes.

## License

MIT
