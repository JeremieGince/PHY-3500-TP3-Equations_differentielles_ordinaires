import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


def resolution_runge_kutta_ordre_2(equation_reduite_ordre_1_gauche, point_initial: float, poitn_final: float,
                                   nombre_de_pas: int, conditions_initiales: float, x=sym.symbols("x"),
                                   t=sym.symbols("t")) -> (float, float, float):
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
    x_0 = conditions_initiales
    t_0 = point_initial
    longueur_de_pas = (poitn_final - point_initial) / nombre_de_pas
    ordre_d_erreur = -int(np.log10(longueur_de_pas**3))
    for step in range(0, nombre_de_pas):
        k1 = longueur_de_pas * equation_reduite_ordre_1_gauche.evalf(ordre_d_erreur, {x: x_0, t: t_0}) / 2
        k2 = longueur_de_pas * equation_reduite_ordre_1_gauche.evalf(ordre_d_erreur, {x: x_0 + k1, t: t_0 + longueur_de_pas / 2})
        t_0 += longueur_de_pas
        x_0 += k2

    erreur_x = 1*(10**(-ordre_d_erreur))
    erreur_v = abs((equation_reduite_ordre_1_gauche.evalf(ordre_d_erreur, {x: x_0 + erreur_x, t: t_0}) -
               equation_reduite_ordre_1_gauche.evalf(ordre_d_erreur, {x: x_0 - erreur_x, t: t_0}))/2)
    return x_0, equation_reduite_ordre_1_gauche.evalf(ordre_d_erreur, {x: x_0, t: t_0}), erreur_x, erreur_v


def afficher_x_ou_v_vs_t_runge_kutta(equation_reduite_ordre_1_gauche, plage: tuple, nombre_de_points: int, conditions_initiales: tuple,
                         afficher_v=False, x=sym.symbols("x"),t=sym.symbols("t") ) -> None:
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
    x_0, v_0 = conditions_initiales
    t_0 = debut
    if afficher_v:
        valeurs_de_v = []
        i = 0
        for t in valeurs_de_t:
            if i == 0:
                valeurs_de_v.append(v_0)
                i += 1
            else:
                x_0, v_0, erreur_x, erreur_v = resolution_runge_kutta_ordre_2(equation_reduite_ordre_1_gauche, t_0, t, 5, 0, x, t)
                valeurs_de_v.append(v_0)
                t_0 = t

        ax.plot(valeurs_de_t, np.asarray(valeurs_de_v), color='blue', lw=2)
        ax.set_title("Graphique illustrant les résultats de l'équation correspondante\n"
             " aux intersections entre les fonctions de gauche et de droite de\n l'équation.")
        ax.set_xlabel("Valeurs de x")
        ax.set_ylabel("Valeurs de f(x) gauche et droite")
    else:
        valeurs_de_x = []
        i = 0
        for t in valeurs_de_t:
            if i == 0:
                valeurs_de_x.append(x_0)
                i += 1
            else:
                x_0, v_0, erreur_x, erreur_v = resolution_runge_kutta_ordre_2(equation_reduite_ordre_1_gauche, t_0, t, 5, 0, x, t)
                valeurs_de_x.append(x_0)
                t_0 = t

        ax.plot(valeurs_de_t, np.asarray(valeurs_de_x), color='blue', lw=2)
        ax.set_title("Graphique illustrant les résultats de l'équation correspondante\n"
             " aux intersections entre les fonctions de gauche et de droite de\n l'équation.")
        ax.set_xlabel("Valeurs de x")
        ax.set_ylabel("Valeurs de f(x) gauche et droite")

    plt.grid()
    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    print("WORK IN PROGRESS")