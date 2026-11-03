"""
Trabalho Prático de Análise Numérica 2026/2
Sistema termodinâmico turbina-compressor-queimador

Bibliotecas: numpy, scipy, sympy.
"""
import numpy as np
import sympy as sp
from scipy.linalg import lu_factor, lu_solve

# ----------------------------------------------------------------------------
# IDENTIFICAÇÃO (preencha antes de gerar os PDFs)
# ----------------------------------------------------------------------------
NOME = ""
MATRICULA = ""

CP = sp.Rational(103, 100)  # 1.03 kJ/(kg K)
Q = 8000
T1 = 25

NOMES = ["w", "p", "Ec", "t2", "Es", "t", "Et"]
X0 = np.array([10.0, 300.0, 1000.0, 250.0, 1500.0, 800.0, 2400.0])
TOLER = 1.0e-10
ITER_MAX = 100

# ----------------------------------------------------------------------------
# Monta F(x) simbolicamente (Equações (1)-(7)) e deriva o Jacobiano com o SymPy
# ----------------------------------------------------------------------------
w, p, Ec, t2, Es, t, Et = sp.symbols("w p Ec t2 Es t Et", real=True)
VARS = [w, p, Ec, t2, Es, t, Et]

R = sp.Rational
f1 = p - (331 + R("45.6") * w - R("4.03") * w ** 2)
f2 = Ec - (1020 - R("0.383") * p + R("0.00513") * p ** 2)
f3 = w - (R("8.5019") + R("0.02332") * p + R("0.48e-4") * p ** 2 - R("0.02644") * t
          + R("0.1849e-4") * t ** 2 + R("0.000121") * p * t - R("0.2736e-6") * p ** 2 * t
          - R("0.1137e-6") * p * t ** 2 + R("0.2124e-9") * p ** 2 * t ** 2)
f4 = Et - (R("1727.5") - R("10.06") * p + R("0.033033") * p ** 2 - R("7.4709") * t
           + R("0.003919") * t ** 2 + R("0.050921") * p * t - R("0.8525e-4") * p ** 2 * t
           - R("0.2356e-4") * p * t ** 2 + R("0.4473e-7") * p ** 2 * t ** 2)
f5 = Ec - w * CP * (t2 - T1)
f6 = Q - w * CP * (t - t2)
f7 = Et - Ec - Es

F_SIM = sp.Matrix([f1, f2, f3, f4, f5, f6, f7])
J_SIM = F_SIM.jacobian(VARS)

# transforma as expressões simbólicas em funções numéricas rápidas
F_NUM = sp.lambdify(VARS, F_SIM, modules="numpy")
J_NUM = sp.lambdify(VARS, J_SIM, modules="numpy")


def F(x):
    return np.array(F_NUM(*x), dtype=float).flatten()


def jacobiano(x):
    return np.array(J_NUM(*x), dtype=float)


def newton_sistema(x0, toler=TOLER, iter_max=ITER_MAX):
    x = x0.copy()
    it = 0
    cond_erro = 1
    print("%4s " % "iter" + " ".join("%14s" % n for n in NOMES)
          + " %13s %13s" % ("||Delta||inf", "||F||inf"))
    print("%4d " % it + " ".join("%14.8f" % v for v in x)
          + " %13s %13.4e" % ("-", np.linalg.norm(F(x), np.inf)))
    while it < iter_max:
        Fx = F(x)
        J = jacobiano(x)
        lu, piv = lu_factor(J)  # decomposição LU com pivotação parcial
        delta = lu_solve((lu, piv), -Fx)  # resolve J*delta = -F (sem inverter J)
        x = x + delta
        it += 1
        norma_delta = np.max(np.abs(delta))
        print("%4d " % it + " ".join("%14.8f" % v for v in x)
              + " %13.4e %13.4e" % (norma_delta, np.linalg.norm(F(x), np.inf)))
        if norma_delta <= toler:
            cond_erro = 0
            break
    return x, it, cond_erro


if __name__ == "__main__":
    print("Trabalho Prático de Análise Numérica 2026/2")
    print("Aluno(a): %s    Matrícula: %s" % (NOME, MATRICULA))
    print("Critério de parada: ||Delta||_inf <= %.0e\n" % TOLER)

    x, iters, cond = newton_sistema(X0)

    print("\nCondErro = %d  (0 = convergiu)   Iterações = %d" % (cond, iters))
    print("\nPonto de operação:")
    unidades = ["kg/s", "kPa", "kW", "°C", "kW", "°C", "kW"]
    for nome, v, u in zip(NOMES, x, unidades):
        print("  %-3s = %16.10f %s" % (nome, v, u))
    print("\n||F(x)||_inf final = %.3e" % np.linalg.norm(F(x), np.inf))
