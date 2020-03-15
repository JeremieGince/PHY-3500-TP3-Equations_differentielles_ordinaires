import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


def resolution_runge_kutta_ordre_2(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, point_initial: float, poitn_final: float,
                                   nombre_de_pas: int, conditions_initiales: tuple, x=sym.symbols("x"),
                                   t=sym.symbols("t"), v=sym.symbols("v")) -> (float, float, float):
    """
    Cette méthode utilise la méthode de résolution d'équation non-linéaire par relaxation.
    Pour être utilisé, l'équation doit être de la forme x = f(x) afin que l'algorithme ne travail
    qu'à évaluer f(x). La méthode redéfinis x_n+1 = f(x_n) jusqu'à ce que l'erreur sur x_n+1
    soit inférieur à l'erreur visée.
    Parameters
    ----------
    fonction : SymPy object
        f(x) à évaluer pour la méthode
    point_initial :
        Valeur initial pour débuter l'application de la méthode
        par relaxation
    erreur_visee :
        Valeur de l'erreur que nous visons pour le résultats de la résolution
        de l'équation
    fonction_derivee : SymPy object
        f'(x) pourcalculer l'évolution de l'erreur sur le résultat. Si ce paramètre
        n'est pas remplis, la fonction f(x) sera dérivée à l'aide de sympy
    x : SymPy symbol
        variable de l'équation à résoudre
    """
    x_0, v_0 = conditions_initiales
    t_0 = point_initial
    longueur_de_pas = (poitn_final - point_initial) / nombre_de_pas
    ordre_d_erreur = -int(np.log10(longueur_de_pas**3))
    for step in range(0, nombre_de_pas):
        k1_v = longueur_de_pas * equation_reduite_ordre_1_v.evalf(ordre_d_erreur, {x: x_0, t: t_0, v: v_0}) / 2
        k2_v = longueur_de_pas * equation_reduite_ordre_1_v.evalf(ordre_d_erreur, {x: x_0, t: t_0 + (longueur_de_pas / 2), v: v_0 + k1_v})
        k1_x = longueur_de_pas * equation_reduite_ordre_1_x.evalf(ordre_d_erreur, {x: x_0, t: t_0, v: v_0}) / 2
        k2_x = longueur_de_pas * equation_reduite_ordre_1_x.evalf(ordre_d_erreur, {x: x_0 + k1_x, t: t_0 + (longueur_de_pas / 2), v:v_0})
        t_0 += longueur_de_pas
        x_0 += k2_x
        v_0 += k2_v
    erreur = 1*(10**(-ordre_d_erreur))
    return x_0, v_0, erreur


def afficher_x_ou_v_vs_t_runge_kutta(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, plage: tuple, nombre_de_points: int, conditions_initiales: tuple,
                         afficher = "x_vs_t", x=sym.symbols("x"),t=sym.symbols("t"), v=sym.symbols("v") ) -> None:
    """
    Cette méthode utilise la méthode de résolution d'équation non-linéaire par relaxation.
    Pour être utilisé, l'équation doit être de la forme x = f(x) afin que l'algorithme ne travail
    qu'à évaluer f(x). La méthode redéfinis x_n+1 = f(x_n) jusqu'à ce que l'erreur sur x_n+1
    soit inférieur à l'erreur visée.
    Parameters
    ----------
    fonction : SymPy object
        f(x) à évaluer pour la méthode
    point_initial :
        Valeur initial pour débuter l'application de la méthode
        par relaxation
    erreur_visee :
        Valeur de l'erreur que nous visons pour le résultats de la résolution
        de l'équation
    fonction_derivee : SymPy object
        f'(x) pourcalculer l'évolution de l'erreur sur le résultat. Si ce paramètre
        n'est pas remplis, la fonction f(x) sera dérivée à l'aide de sympy
    x : SymPy symbol
        variable de l'équation à résoudre
    """
    debut, fin = plage
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    valeurs_de_t = np.linspace(debut, fin, nombre_de_points)
    x_i, v_i = conditions_initiales
    t_i = debut
    valeurs_de_v = []
    valeurs_de_x = []
    i = 0
    for temps in valeurs_de_t:
        if i == 0:
            valeurs_de_v.append(v_i)
            valeurs_de_x.append(x_i)
            i = 1
        else:
            x_i, v_i, erreur = resolution_runge_kutta_ordre_2(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, t_i, temps, 10, (x_i, v_i), x, t, v)
            valeurs_de_v.append(v_i)
            valeurs_de_x.append(x_i)
            t_i = temps

    if afficher == "x_vs_t":
        ax.plot(valeurs_de_t, np.asarray(valeurs_de_v), color='blue', lw=2)
        ax.set_title("Graphique illustrant la vitesse en fonction du temps")
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Vitesse [m/s]")

    elif afficher == "v_vs_t":
        ax.plot(valeurs_de_t, np.asarray(valeurs_de_x), color='blue', lw=2)
        ax.set_title("Graphique illustrant la position en fonction du temps")
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Position [m]")

    elif afficher == "v_vs_x":
        ax.plot(np.asarray(valeurs_de_x), np.asarray(valeurs_de_v), color='blue', lw=2)
        ax.set_title("Graphique illustrant la vitesse en fonction de la position")
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Vitesse [m/s]")

    plt.grid()
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Code pour la question a)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -x
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (1, 0))
    """
    # Code pour la question b)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -x
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (3, 0))
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (-2, 0))
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (0, 2))
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (0, -3))
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (2, 2))
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (-2, -2))
    """
    # Code pour la question c)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -(x**3)
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (1, 0))
    """
    # Code pour la question d)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -(x**3)
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (1, 0), "v_vs_x")
    equation_v = -x
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 101, (1, 0))
    """