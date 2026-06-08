from scipy.optimize import fsolve

def func(uvw, Pu, Pv, delta_u, delta_v, delta_w, k_on, k_off, alpha):
    u = uvw[0]
    v = uvw[1]
    w = uvw[2]
    return [
        Pu + alpha * w - delta_u * u - 2 * k_on * u**2 * v + 2 * k_off * w,
        Pv             - delta_v * v -     k_on * u**2 * v +     k_off * w,
        - delta_w * w                +     k_on * u**2 * v -     k_off * w
    ]

def get_steady_states(Pu, Pv, delta_u, delta_v, delta_w, k_on, k_off, alpha):
    guess = [0.910, 0.959, 0.795]
    root = fsolve(func, guess, args=(Pu, Pv, delta_u, delta_v, delta_w, k_on, k_off, alpha))
    return root