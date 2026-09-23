"""Example uses of the physica DSL.

Run with:  python3 example_physics.py
"""

import sympy as sp
from physica import Object2, Vector2, TimeVector, Time, T, x, Func, Lim, continuous, impulse, update, simulate, combine, describe_system


# ---------------------------------------------------------------------------
# 1. Projectile with gravity, a steady wind, and an impulsive kick
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Projectile: gravity + wind + kick")
print("=" * 60)

ball = Object2("ball", mass=0.45, position=Vector2(0, 10), velocity=Vector2(5, 0))

@continuous(ball)
class Gravity:
    gravity = Vector2(0, -9.81)

@continuous(ball)
class Wind:
    wind = Vector2(2, 0)

@impulse(ball)
class Kick:
    force = Vector2(50, 20)

@update(ball)
def physics(obj, dt):
    obj.integrate(dt)
    obj.clear_impulses()

for _ in range(100):
    physics(ball, 0.016)

print(f"final position: {ball.position}")
print(f"speed: {ball.speed()}, KE: {ball.kinetic_energy()}")
print()


# ---------------------------------------------------------------------------
# 2. Time-varying force (sinusoidal drive) + closed form via sympy
# ---------------------------------------------------------------------------
print("=" * 60)
print("2. Time-varying drive + symbolic combination")
print("=" * 60)

car = Object2("car", mass=2.0)

@continuous(car)
class Road:
    push = Vector2(1.5, 0)

@continuous(car)
class Engine:
    drive = TimeVector.of(lambda t: Vector2(4 * sp.sin(t), 0))

print(car.describe())
print("\nclosed form (dsolve):")
for solution in car.closed_form():
    print("  ", solution)
print()


# ---------------------------------------------------------------------------
# 3. Multiple objects combined into one system
# ---------------------------------------------------------------------------
print("=" * 60)
print("3. Multi-object system")
print("=" * 60)

rocket = Object2("rocket", mass=50.0)
booster = Object2("booster", mass=10.0)

@continuous(rocket)
class Thrust:
    force = TimeVector(0, 120.0 - 3 * T)   # thrust falls off with time

@continuous(booster)
class Boost:
    force = Vector2(0, 30)

print(describe_system([rocket, booster]))
print()


# ---------------------------------------------------------------------------
# 4. Using a shared clock + simulate(), then replay the history
# ---------------------------------------------------------------------------
print("=" * 60)
print("4. simulate() with a shared clock and history")
print("=" * 60)

clock = Time()
probe = Object2("probe", mass=1.0, position=Vector2(0, 5))

@continuous(probe)
class Spring:
    restoring = TimeVector.of(lambda t: Vector2(0, -2 * probe.position.y))

history = simulate(probe, dt=0.1, length=10, time=clock)
states = list(history.values())
print(f"clock.t = {clock.t}")
print(f"steps recorded = {len(history)}")
print("first state:", states[0][0]["position"])
print("last  state:", states[-1][0]["position"])
print()


# ---------------------------------------------------------------------------
# 5. Bootstrap syntax (Func/Lim) applied to the derived equations of motion
# ---------------------------------------------------------------------------
print("=" * 60)
print("5. Func bootstrap over the physics formulae")
print("=" * 60)

block = Object2("block", mass=2.0)

@continuous(block)
class Gravity:
    gravity = Vector2(0, -9.81)

@continuous(block)
class Ramp:
    push = TimeVector.of(lambda t: Vector2(2 * t, 0))

ax, ay = block.acceleration_func()
print(f"ax(T) = {ax}   ay(T) = {ay}")

# Pretty syntax on the T-bound Funcs
print(f"vx(T)  = ax[-1]   = {ax[-1]}")
print(f"x(T)   = ax[-2]   = {ax[-2]}")
print(f"jerk   = ax[1]    = {ax[1]}")
print(f"ax solves zero at T = {ax.solve()}")
print(f"average ax over [0, 3] = {ax[[0, 3]] / 3}")

# The classic Func(x) behaviour is untouched
f = Func(x**2 - 1)
print(f"Func(x**2-1)(3) = {f(3)},  f[1] = {f[1]},  f[[0,1]] = {f[[0,1]]}")
print(f"Lim(sin(x)/x, x, 0) = {Lim(sp.sin(x)/x, x, 0)}")
print()


# ---------------------------------------------------------------------------
# 6. Pendulum — using the angular DOF (angle / angular_velocity / torque)
# ---------------------------------------------------------------------------
print("=" * 60)
print("6. Pendulum")
print("=" * 60)

import math

L, g, m = 1.0, 9.81, 0.5          # rod length, gravity, bob mass

bob = Object2("bob", mass=m)
bob.moment = m * L ** 2           # I = mL^2 for a point mass on a massless rod
bob.angle = math.pi / 2           # start horizontal
bob.angular_velocity = 0.0

@update(bob)
def pendulum(obj, dt):
    obj.torque = -m * g * L * math.sin(obj.angle)   # restoring torque
    obj.integrate(dt)
    obj.position = Vector2(L * math.sin(obj.angle), -L * math.cos(obj.angle))

def energy(o):
    return 0.5 * o.moment * o.angular_velocity ** 2 - m * g * L * math.cos(o.angle)

E0 = float(energy(bob))
print(f"E(0) = {E0:.6g}")
for _ in range(200):
    pendulum(bob, 0.01)

print(f"angle = {float(bob.angle):.6f} rad")
print(f"omega = {float(bob.angular_velocity):.6f} rad/s")
# The angular DOF uses an explicit update under every scheme, so the energy
# drift here is first-order in dt: halving dt halves the drift.
print(f"E(t)  = {float(energy(bob)):.6g}  (E0 = {E0:.6g}; drift is O(dt))")
print(f"bob position (x, y) = {bob.position}")
print()


# ---------------------------------------------------------------------------
# 7. Pendulum, small-angle approximation  sin(theta) ~ theta
# ---------------------------------------------------------------------------
print("=" * 60)
print("7. Pendulum (small-angle: sin(theta) ~ theta)")
print("=" * 60)

theta0 = 0.1
small = Object2("small", mass=m)
small.moment = m * L ** 2
small.angle = theta0
small.angular_velocity = 0.0

@update(small)
def small_pendulum(obj, dt):
    obj.torque = -m * g * L * obj.angle            # linear restoring torque
    obj.integrate(dt)
    obj.position = Vector2(L * math.sin(obj.angle), -L * math.cos(obj.angle))

omega0 = math.sqrt(g / L)
dt = 0.0005
max_err = 0.0
for i in range(1, 4001):
    small_pendulum(small, dt)
    t = i * dt
    analytic = theta0 * math.cos(omega0 * t)       # exact SHM solution
    max_err = max(max_err, abs(float(small.angle) - analytic))

print(f"omega = sqrt(g/L) = {omega0:.6f} rad/s")
print(f"period = 2*pi/omega = {2*math.pi/omega0:.6f} s")
print(f"amplitude-independent (SHM), unlike the exact sin(theta) pendulum")
print(f"max |numeric - {theta0}*cos(omega*t)| over 2 s = {max_err:.2e}")


