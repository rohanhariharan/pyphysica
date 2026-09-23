"""Tests for pyphysica.

Covers the symbolic (Func/Lim), vector, and physics (Object2/TimeVector)
layers, including regression checks for the core numeric integrator.
"""

import math

import pytest
import sympy as sp

from physica import (
    Object2,
    Vector2,
    TimeVector,
    Time,
    T,
    x,
    Func,
    Lim,
    continuous,
    impulse,
    update,
    simulate,
    combine,
    describe_system,
    INTEGRATORS,
    integrate_verlet,
    integrate_euler_cromer,
    integrate_rk4,
)


# ---------------------------------------------------------------------------
# Func / Lim
# ---------------------------------------------------------------------------


def test_func_default_variable_is_x():
    f = Func(x**2 - 1)
    assert f(3) == 8
    assert str(f[1]) == "2*x"


def test_func_variable_aware():
    g = Func(3 * T + 1, var=T)
    assert g(2) == 7
    assert str(g[1]) == "3"


def test_func_arithmetic_preserves_variable():
    a = Func(T**2, var=T)
    b = Func(T + 1, var=T)
    assert sp.simplify((a + b).expr - (T**2 + T + 1)) == 0
    assert sp.simplify((a * b).expr - (T**2 * (T + 1))) == 0
    assert sp.simplify((a / b).expr - (T**2 / (T + 1))) == 0
    assert (a + b).var == T


def test_func_derivative_and_integral():
    f = Func(x**3)
    assert str(f.derivative(1).expr) == "3*x**2"
    assert str(f.integrate(1).expr) == "x**4/4"


def test_func_definite_integral():
    f = Func(x**2)
    assert f[[0, 1]] == sp.Rational(1, 3)


def test_func_composition():
    f = Func(x**2)
    g = Func(x + 1)
    assert sp.simplify((f @ g).expr - (x + 1) ** 2) == 0


def test_func_eq():
    assert Func(x**2 + 2 * x + 1) == Func((x + 1) ** 2)


def test_lim():
    assert Lim(sp.sin(x) / x, x, 0) == 1


# ---------------------------------------------------------------------------
# Vector2
# ---------------------------------------------------------------------------


def test_vector_ops():
    a = Vector2(3, 4)
    b = Vector2(1, 2)
    assert a.norm() == 5
    assert float(a.dot(b)) == 11
    assert (a + b) == Vector2(4, 6)
    assert (a - b) == Vector2(2, 2)
    assert float(a.cross(b)) == 2


def test_vector_normalized():
    n = Vector2(3, 4).normalized()
    assert abs(float(n.norm()) - 1) < 1e-12


def test_vector_normalize_zero_raises():
    with pytest.raises(ZeroDivisionError):
        Vector2(0, 0).normalized()


# ---------------------------------------------------------------------------
# TimeVector
# ---------------------------------------------------------------------------


def test_timevector_symbolic_and_at():
    tv = TimeVector(0, 120.0 - 3 * T)
    assert float(tv.at(10).y) == 90.0
    assert sp.simplify(tv.symbolic().y - (120.0 - 3 * T)) == 0


def test_timevector_of():
    tv = TimeVector.of(lambda t: Vector2(4 * sp.sin(t), 0))
    assert abs(float(tv.at(math.pi / 2).x) - 4.0) < 1e-12


def test_timevector_copy_is_independent():
    tv = TimeVector(1, 2)
    cp = tv.copy()
    assert cp is not tv
    assert float(cp.x) == 1 and float(cp.y) == 2
    assert float(tv.x) == 1


def test_timevector_components_are_readonly():
    tv = TimeVector(1, 2)
    with pytest.raises(AttributeError):
        tv.x = 99


# ---------------------------------------------------------------------------
# Object2 force management
# ---------------------------------------------------------------------------


def test_continuous_force_decorator():
    o = Object2("o", mass=1.0)

    @continuous(o)
    class Gravity:
        f = Vector2(0, -9.81)

    assert len(o.continuous_forces) == 1
    assert float(o.net_force.y) == -9.81


def test_continuous_forces_is_live_property():
    """Force lists must reflect late additions (regression: stale snapshot)."""
    o = Object2("o", mass=1.0)
    o.apply_force_continuous(Vector2(5, 0))
    assert float(o.net_force.x) == 5.0
    o.apply_force_continuous(Vector2(0, 3))
    assert float(o.net_force.y) == 3.0


def test_impulse_changes_velocity():
    o = Object2("o", mass=2.0)
    o.apply_force_impulse(Vector2(4, 0))
    assert float(o.velocity.x) == 2.0


def test_invalid_force_type_raises():
    o = Object2("o", mass=1.0)
    with pytest.raises(TypeError):
        o.apply_force_continuous("not a force")


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def test_projectile_kinematics():
    """Constant force reproduces the closed-form constant-acceleration result."""
    o = Object2("o", mass=1.0, position=Vector2(0, 10), velocity=Vector2(5, 0))

    @continuous(o)
    class Gravity:
        f = Vector2(0, -9.81)

    dt = 0.001
    steps = 1000
    for _ in range(steps):
        o.integrate(dt)
    t = steps * dt
    assert abs(float(o.velocity.x) - 5.0) < 1e-9
    assert abs(float(o.velocity.y) - (-9.81 * t)) < 1e-6
    assert abs(float(o.position.y) - (10 - 0.5 * 9.81 * t**2)) < 1e-3


def test_time_varying_force_matches_analytic():
    """a_y = 10 - 2t  =>  v(t) = 10t - t**2,  y(t) = 5t**2 - t**3/3."""
    o = Object2("o", mass=1.0)

    @continuous(o)
    class Drive:
        f = TimeVector(0, 10 - 2 * T)

    dt = 0.001
    for _ in range(2000):
        o.integrate(dt)

    t = 2.0
    assert abs(float(o.velocity.y) - (10 * t - t**2)) < 0.01
    assert abs(float(o.position.y) - (5 * t**2 - t**3 / 3)) < 0.02


def test_object_time_advances():
    o = Object2("o", mass=1.0)
    for _ in range(100):
        o.integrate(0.01)
    assert abs(o.time - 1.0) < 1e-9


def test_simulate_records_history():
    o = Object2("o", mass=1.0, position=Vector2(0, 5))
    history = simulate(o, dt=0.1, length=10)
    assert len(history) == 10
    assert all(isinstance(v, list) and len(v) == 1 for v in history.values())


def test_update_decorator_function_form():
    o = Object2("o", mass=1.0, velocity=Vector2(1, 0))

    @update(o)
    def physics(obj, dt):
        obj.integrate(dt)

    for _ in range(10):
        physics(o, 0.1)
    assert abs(float(o.position.x) - 1.0) < 1e-9
    assert len(physics.history) == 10


# ---------------------------------------------------------------------------
# Symbolic physics
# ---------------------------------------------------------------------------


def test_equation_of_motion():
    o = Object2("o", mass=2.0)

    @continuous(o)
    class Drive:
        f = TimeVector(0, 4 * T)

    ax, ay = o.equation_of_motion()
    assert sp.simplify(ax) == 0
    assert sp.simplify(ay - 2 * T) == 0


def test_acceleration_func_is_t_bound():
    o = Object2("o", mass=2.0)

    @continuous(o)
    class Gravity:
        f = Vector2(0, -9.81)

    ax, ay = o.acceleration_func()
    assert ax.var == T and ay.var == T
    assert abs(float(ay(1)) + 9.81 / 2) < 1e-12


def test_velocity_and_position_funcs():
    """a_y = -10, from rest => v_y = -10T, y = -5T**2."""
    o = Object2("o", mass=1.0)

    @continuous(o)
    class Gravity:
        f = Vector2(0, -10)

    vy = o.velocity_func()[1]
    py = o.position_func()[1]
    assert sp.simplify(vy.expr + 10 * T) == 0
    assert sp.simplify(py.expr + 5 * T**2) == 0


def test_closed_form_returns_two_solutions():
    o = Object2("o", mass=1.0)

    @continuous(o)
    class Drive:
        f = Vector2(0, 6.0)

    sols = o.closed_form()
    assert len(sols) == 2


def test_describe_and_describe_system():
    a = Object2("a", mass=1.0)

    @continuous(a)
    class Fa:
        f = Vector2(1, 0)

    b = Object2("b", mass=3.0)

    @continuous(b)
    class Fb:
        f = Vector2(0, -3)

    assert "a" in a.describe()
    text = describe_system([a, b])
    assert "System of 2 object(s)" in text


def test_combine():
    a = Object2("a", mass=1.0)

    @continuous(a)
    class Fa:
        f = Vector2(1, 0)

    b = Object2("b", mass=3.0)

    @continuous(b)
    class Fb:
        f = Vector2(0, -3)

    system = combine([a, b])
    assert float(system["mass"]) == 4.0
    assert float(system["net_force"].x) == 1.0
    assert float(system["net_force"].y) == -3.0
    assert len(system["accelerations"]) == 2


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", sorted(INTEGRATORS))
def test_all_integrators_handle_constant_force(scheme):
    """All schemes reproduce constant-acceleration motion to their own accuracy.

    Verlet and RK4 are exact for constant force. Euler-Cromer is first-order:
    it shifts position by the known half-step offset a*dt*t/2.
    """
    o = Object2("o", mass=1.0, position=Vector2(0, 0), velocity=Vector2(0, 0), integrator=scheme)

    @continuous(o)
    class G:
        f = Vector2(0, -10)

    dt, steps = 0.001, 1000
    for _ in range(steps):
        o.integrate(dt)
    t = steps * dt

    tolerance = 1e-9 if scheme != "euler-cromer" else 0.5 * 10 * dt * t + 1e-9
    assert abs(float(o.position.y) - (-5.0)) < tolerance
    assert abs(float(o.velocity.y) - (-10.0)) < 1e-9


def test_integrator_selection_by_name_and_callable():
    o = Object2("o", mass=1.0)

    @continuous(o)
    class G:
        f = Vector2(0, -10)

    o.integrate(0.1, "rk4")
    assert abs(o.time - 0.1) < 1e-12

    o.integrate(0.1, integrate_verlet)
    assert abs(o.time - 0.2) < 1e-12

    with pytest.raises(ValueError):
        o.integrate(0.1, "does-not-exist")

    with pytest.raises(TypeError):
        o.integrate(0.1, 123)


def _spring_energy(scheme, steps=2000, dt=0.01):
    k, m = 4.0, 1.0
    o = Object2("o", mass=m, position=Vector2(1.0, 0), velocity=Vector2(0, 0), integrator=scheme)

    @continuous(o)
    class Spring:
        f = TimeVector.of(lambda t: Vector2(-k * o.position.x, 0))

    def energy():
        return 0.5 * m * float(o.velocity.x) ** 2 + 0.5 * k * float(o.position.x) ** 2

    e0 = energy()
    worst = 0.0
    for _ in range(steps):
        o.integrate(dt)
        worst = max(worst, abs(energy() - e0))
    return worst


def test_verlet_is_symplectic_bounded_energy():
    """Velocity Verlet keeps energy bounded (no secular drift) on a spring."""
    assert _spring_energy("verlet") < 1e-2


def test_rk4_is_more_accurate_than_verlet():
    """RK4 should have far smaller energy error at the same dt."""
    assert _spring_energy("rk4") < _spring_energy("verlet")


def _spring_pos_error(scheme, dt):
    """|x(2) - cos(4)| for the k=4, m=1 spring released from x=1."""
    k, m = 4.0, 1.0
    o = Object2("o", mass=m, position=Vector2(1.0, 0), velocity=Vector2(0, 0), integrator=scheme)

    @continuous(o)
    class Spring:
        f = TimeVector.of(lambda t: Vector2(-k * o.position.x, 0))

    for _ in range(round(2.0 / dt)):
        o.integrate(dt)
    return abs(float(o.position.x) - math.cos(4))


def test_integrator_order_of_accuracy():
    """Halving dt: Verlet error / 4 (2nd order), RK4 error / 16 (4th order)."""
    v_ratio = _spring_pos_error("verlet", 0.02) / _spring_pos_error("verlet", 0.01)
    r_ratio = _spring_pos_error("rk4", 0.02) / _spring_pos_error("rk4", 0.01)
    assert 3.0 < v_ratio < 5.0
    assert r_ratio > 12.0


def test_rk4_restores_state_after_staging():
    """RK4 stages the object; final state must not be left mid-stage."""
    o = Object2("o", mass=1.0, position=Vector2(0, 0), integrator="rk4")

    @continuous(o)
    class G:
        f = Vector2(0, -10)

    o.integrate(0.1)
    assert abs(float(o.position.y) + 0.05) < 1e-9


# ---------------------------------------------------------------------------
# Known-answer physics checks
# ---------------------------------------------------------------------------


def test_small_angle_pendulum_period():
    """Numeric SHM tracks theta0*cos(omega*t) with omega = sqrt(g/L)."""
    L, g, m = 1.0, 9.81, 0.5
    theta0 = 0.1
    bob = Object2("bob", mass=m)
    bob.moment = m * L**2
    bob.angle = theta0
    bob.angular_velocity = 0.0

    @update(bob)
    def pendulum(obj, dt):
        obj.torque = -m * g * L * obj.angle
        obj.integrate(dt)

    omega0 = math.sqrt(g / L)
    assert abs(2 * math.pi / omega0 - 2.006067) < 1e-6

    dt = 0.0005
    max_err = 0.0
    for i in range(1, 4001):
        pendulum(bob, dt)
        analytic = theta0 * math.cos(omega0 * i * dt)
        max_err = max(max_err, abs(float(bob.angle) - analytic))
    assert max_err < 1e-3
