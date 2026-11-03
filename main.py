"""
------------------------------------------------------------------------------------------------------------------------
Trabalho Prático de Análise Numérica 2026/2
Sistema termodinâmico turbina-compressor-queimador

Newton para sistemas com JACOBIANO NUMÉRICO e decomposição LU com pivotação parcial.

Bibliotecas: numpy
------------------------------------------------------------------------------------------------------------------------
"""
import numpy as np

# IDENTIFICAÇÃO
NOME = ""
MATRICULA = ""

# Constantes do problema
CP = 1.03          # kJ/(kg K)
Q = 8000.0         # kW (energia adicionada no queimador)
T1 = 25.0          # °C (temperatura de admissão do compressor)

NOMES = ["w", "p", "Ec", "t2", "Es", "t", "Et"]
X0 = np.array([10.0, 300.0, 1000.0, 250.0, 1500.0, 800.0, 2400.0])

TOLER = 1.0e-10    # critério: ||Delta||_inf <= 1e-10
ITER_MAX = 100


# Sistema não linear F(x) = 0  (Equações (1)-(7) escritas na forma F_i(x) = 0)
def F(x):
    w, p, Ec, t2, Es, t, Et = x
    f = np.zeros(7)
    # (1) p = 331 + 45.6 w - 4.03 w^2
    f[0] = p - (331.0 + 45.6 * w - 4.03 * w**2)
    # (2) Ec = 1020 - 0.383 p + 0.00513 p^2
    f[1] = Ec - (1020.0 - 0.383 * p + 0.00513 * p**2)
    # (3) w = g(p, t)
    f[2] = w - (8.5019 + 0.02332 * p + 0.48e-4 * p**2 - 0.02644 * t
                + 0.1849e-4 * t**2 + 0.000121 * p * t - 0.2736e-6 * p**2 * t
                - 0.1137e-6 * p * t**2 + 0.2124e-9 * p**2 * t**2)
    # (4) Et = h(p, t)
    f[3] = Et - (1727.5 - 10.06 * p + 0.033033 * p**2 - 7.4709 * t
                 + 0.003919 * t**2 + 0.050921 * p * t - 0.8525e-4 * p**2 * t
                 - 0.2356e-4 * p * t**2 + 0.4473e-7 * p**2 * t**2)
    # (5) Ec = w cp (t2 - 25)
    f[4] = Ec - w * CP * (t2 - T1)
    # (6) 8000 = w cp (t - t2)
    f[5] = Q - w * CP * (t - t2)
    # (7) Et = Ec + Es
    f[6] = Et - Ec - Es
    return f


# Matriz Jacobiana numérica por diferenças finitas centrais:
#   dF_i/dx_j  ~  [F_i(x + h_j e_j) - F_i(x - h_j e_j)] / (2 h_j)
# com passo relativo h_j = H_REL * max(1, |x_j|).
# Erro de truncamento O(h^2)
H_REL = 1.0e-6


def jacobiano(x):
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        h = H_REL * max(1.0, abs(x[j]))
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (F(xp) - F(xm)) / (2.0 * h)
    return J


# (LU é escrita sobre A; pivot guarda a permutação das linhas)
def decomposicao_lu(A):
    n = A.shape[0]
    A = A.copy()
    pivot = list(range(n))
    det = 1.0
    for j in range(n - 1):
        # escolha do elemento pivô
        p = j
        amax = abs(A[j, j])
        for k in range(j + 1, n):
            if abs(A[k, j]) > amax:
                amax = abs(A[k, j])
                p = k
        if p != j:
            # troca de linhas
            for k in range(n):
                A[j, k], A[p, k] = A[p, k], A[j, k]
            pivot[j], pivot[p] = pivot[p], pivot[j]
            det = -det
        det = det * A[j, j]
        if abs(A[j, j]) != 0.0:
            # eliminação de Gauss
            r = 1.0 / A[j, j]
            for i in range(j + 1, n):
                mult = A[i, j] * r
                A[i, j] = mult
                for k in range(j + 1, n):
                    A[i, k] = A[i, k] - mult * A[j, k]
    det = det * A[n - 1, n - 1]
    return A, det, pivot


# Substituições sucessivas pivotais (resolve L y = P b)
def substituicoes_sucessivas_pivotal(L, b, pivot):
    n = L.shape[0]
    y = np.zeros(n)
    k = pivot[0]
    y[0] = b[k]
    for i in range(1, n):
        soma = 0.0
        for j in range(i):
            soma = soma + L[i, j] * y[j]
        k = pivot[i]
        y[i] = b[k] - soma
    return y


# Substituições retroativas (resolve U x = y)
def substituicoes_retroativas(U, y):
    n = U.shape[0]
    x = np.zeros(n)
    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma = soma + U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]
    return x


def resolve_sistema_linear(A, b):
    """Resolve A d = b via decomposição LU """
    LU, det, pivot = decomposicao_lu(A)
    if det == 0.0:
        raise ZeroDivisionError("Jacobiano singular")
    y = substituicoes_sucessivas_pivotal(LU, b, pivot)
    return substituicoes_retroativas(LU, y)


# Método de Newton para sistemas:  J(x_k) Delta = -F(x_k);  x_{k+1} = x_k + Delta
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
        delta = resolve_sistema_linear(J, -Fx)   # resolve J*delta = -F via LU
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
    # Essa parte serve para gerar o PDF com os resultaddos
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
