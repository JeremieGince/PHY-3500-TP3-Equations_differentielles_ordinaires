import sympy as sym
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint
import scipy.optimize as sciopt
from mpl_toolkits import mplot3d


def resolution_runge_kutta_ordre_2(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, point_initial: float, poitn_final: float,
                                   nombre_de_pas: int, conditions_initiales: tuple, x=sym.symbols("x"),
                                   t=sym.symbols("t"), v=sym.symbols("v")) -> (float, float, float):
    """
    Cette méthode résolve une équation différentielle ordinaire de degrée 2 en effectuant
    un changement de variable dx/dt = v. La méthode de résolution utilise runge kutta d'ordre 2
    pour les deux équations différentielles produites par le changement de variable.

    Parameters
    ----------
    equation_reduite_ordre_1_x : SymPy object
        fontion associé à dx/dt ex: dx/dt = v
    equation_reduite_ordre_1_v : SymPy object
        fontion associé à dv/dt ex: dv/dt =  -w^2x + sin(t)
    point_initial :
        Valeur initial associé à la valeur indépendante de l'équation différentiel
    poitn_final :
        Valeur finnale de la variable indépendante. Correspond à la valeur
        indépendante associé aux résultats des deux variables dépendantes retournées
    nombre_de_pas :
        Nombre de pas effectué par la méthode runge kuta
    conditions_initiales:
        tuple des conditions initiales pour
        les deux variables dépendantes ex (x_0, v_0)
    x : SymPy symbol
        variable dépendante d'ordre 1
    t : SymPy symbol
        variable indépendante
    v : SymPy symbol
        variable dépendante d'ordre 2

    Returns
    -----------
    x_0:
        Valeur finnale de la variable dépendante d'ordre 1

    v_0:
        Valeur finnale de la variable dépendante d'ordre 2

    erreur:
        erreur sur les deux résultats
    """
    x_0, v_0 = conditions_initiales
    t_0 = point_initial
    longueur_de_pas = (poitn_final - point_initial) / nombre_de_pas
    ordre_d_erreur = -int(np.log10(longueur_de_pas**3))
    for step in range(0, nombre_de_pas):
        k1_v = longueur_de_pas * equation_reduite_ordre_1_v.evalf(ordre_d_erreur, {x: x_0, t: t_0, v: v_0}) / 2
        k2_v = longueur_de_pas * equation_reduite_ordre_1_v.evalf(ordre_d_erreur, {x: x_0, t: t_0 + (longueur_de_pas / 2), v: v_0 + k1_v})
        k1_x = longueur_de_pas * equation_reduite_ordre_1_x.evalf(ordre_d_erreur, {x: x_0, t: t_0, v: v_0}) / 2
        k2_x = longueur_de_pas * equation_reduite_ordre_1_x.evalf(ordre_d_erreur, {x: x_0 + k1_x, t: t_0 + (longueur_de_pas / 2), v: v_0})
        t_0 += longueur_de_pas
        x_0 += k2_x
        v_0 += k2_v
    erreur = 1*(10**(-ordre_d_erreur))
    return x_0, v_0, erreur


def afficher_x_ou_v_vs_t_runge_kutta(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, plage: tuple, nombre_de_points: int, conditions_initiales: tuple,
                                            afficher ="x_vs_t", frequence_comparative =0, x=sym.symbols("x"), t=sym.symbols("t"), v=sym.symbols("v")) -> None:
    """
    Cette méthode appel la resolution_runge_kutta_ordre_2 pour produire les points pour une certaine plage donnée

    Parameters
    ----------
    equation_reduite_ordre_1_x : SymPy object
        fontion associé à dx/dt ex: dx/dt = v
    equation_reduite_ordre_1_v : SymPy object
        fontion associé à dv/dt ex: dv/dt =  -w^2x + sin(t)
    plage :
        plage d'affichage pour le graphique
    nombre_de_points:
        Nombre de points désirés pour produire le graphique
    conditions_initiales:
        tuple des conditions initiales pour
        les deux variables dépendantes ex (x_0, v_0)
    afficher:
        selon le string donnée
        (x_vs_t, v_vs_t ou v_vsx)
        on affiche un variable en fonction de l'autre
    frequence_comparative:
        si ce paramètre est non null, nous affichons
        aussi un sinus ayant la fréquence donnée
    x : SymPy symbol
        variable dépendante d'ordre 1
    t : SymPy symbol
        variable indépendante
    v : SymPy symbol
        variable dépendante d'ordre 2
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
        ax.plot(valeurs_de_t, np.asarray(valeurs_de_x), color='blue', lw=2)
        ax.set_title("Graphique illustrant la position en fonction du temps")
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Position [m]")

    elif afficher == "v_vs_t":
        ax.plot(valeurs_de_t, np.asarray(valeurs_de_v), color='blue', lw=2)
        ax.set_title("Graphique illustrant la vitesse en fonction du temps")
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Vitesse [m/s]")

    elif afficher == "v_vs_x":
        ax.plot(np.asarray(valeurs_de_x), np.asarray(valeurs_de_v), color='blue', lw=2)
        ax.set_title("Graphique illustrant la vitesse en fonction de la position")
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Vitesse [m/s]")

    if frequence_comparative != 0:
        fonction = lambda time : np.sin(frequence_comparative*time)
        ax.plot(valeurs_de_t, fonction(valeurs_de_t), color='red', lw=2, label=f"sinus ayant une férquence angulaire de {frequence_comparative}rad/s")
        ax.legend()

    plt.grid()
    plt.show()
    plt.close(fig)


def resolution_runge_kutta_ordre_2_scipy(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, point_initial: float, poitn_final: float,
                                                                                        conditions_initiales: tuple, x=sym.symbols("x"),
                                                                    t=sym.symbols("t"), v=sym.symbols("v")):
    """
    Cette méthode résolve une équation différentielle ordinaire de degrée 2 en effectuant
    un changement de variable dx/dt = v. La méthode de résolution utilise runge kutta d'ordre 2
    pour les deux équations différentielles produites par le changement de variable.
    Puis cette méthode appel l'algorithme scipy pour effectuer runge kutta 45.

    Parameters
    ----------
    equation_reduite_ordre_1_x : SymPy object
        fontion associé à dx/dt ex: dx/dt = v
    equation_reduite_ordre_1_v : SymPy object
        fontion associé à dv/dt ex: dv/dt =  -w^2x + sin(t)
    point_initial :
        Valeur initial associé à la valeur indépendante de l'équation différentiel
    poitn_final :
        Valeur finnale de la variable indépendante. Correspond à la valeur
        indépendante associé aux résultats des deux variables dépendantes retournées
    conditions_initiales:
        tuple des conditions initiales pour
        les deux variables dépendantes ex (x_0, v_0)
    x : SymPy symbol
        variable dépendante d'ordre 1
    t : SymPy symbol
        variable indépendante
    v : SymPy symbol
        variable dépendante d'ordre 2

    Returns
    -----------
    solution: OdeObject de scipy
        Un objet scipy contenant toues les
        infos sur le résultat
    """
    x_0, v_0 = conditions_initiales
    t_0 = point_initial

    def fusion_deux_equations(temps, x_et_v):
        valeur_x = x_et_v[0]
        valeur_v = x_et_v[1]
        fv = equation_reduite_ordre_1_v.evalf(9, {t: temps, x: valeur_x, v: valeur_v})
        fx = equation_reduite_ordre_1_x.evalf(9, {t: temps, x: valeur_x, v: valeur_v})

        return np.array([fx, fv], float)

    y_0 = np.array([x_0, v_0], float)
    solution = scint.solve_ivp(fusion_deux_equations, (t_0, poitn_final), y_0, method="RK45", dense_output=True)
    return solution


def afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, plage: tuple, nombre_de_points: int, conditions_initiales: tuple,
                         afficher="x_vs_t", frequence_comparative=0, x=sym.symbols("x"), t=sym.symbols("t"), v=sym.symbols("v")) -> None:
    """
    Cette méthode appel la resolution_runge_kutta_ordre_2_scipy pour produire les points pour une certaine plage donnée

    Parameters
    ----------
    equation_reduite_ordre_1_x : SymPy object
        fontion associé à dx/dt ex: dx/dt = v
    equation_reduite_ordre_1_v : SymPy object
        fontion associé à dv/dt ex: dv/dt =  -w^2x + sin(t)
    plage :
        plage d'affichage pour le graphique
    nombre_de_points:
        Nombre de points désirés pour produire le graphique
    conditions_initiales:
        tuple des conditions initiales pour
        les deux variables dépendantes ex (x_0, v_0)
    afficher:
        selon le string donnée
        (x_vs_t, v_vs_t, v_vs_x ou 3d)
        on affiche un variable en fonction de l'autre
    frequence_comparative:
        si ce paramètre est non null, nous affichons
        aussi un sinus ayant la fréquence donnée
    x : SymPy symbol
        variable dépendante d'ordre 1
    t : SymPy symbol
        variable indépendante
    v : SymPy symbol
        variable dépendante d'ordre 2
    """
    debut, fin = plage
    fig = plt.figure()

    rk45 = resolution_runge_kutta_ordre_2_scipy(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v,  debut, fin, conditions_initiales, x=x, t=t, v=v)
    valeurs_t = np.linspace(debut, fin, nombre_de_points)

    if afficher == "x_vs_t":
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(valeurs_t, rk45.sol(valeurs_t)[0], color='blue', lw=2)
        ax.set_title("Graphique illustrant la position en fonction du temps")
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Position [m]")

    elif afficher == "v_vs_t":
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(valeurs_t, rk45.sol(valeurs_t)[1], color='blue', lw=2)
        ax.set_title("Graphique illustrant la vitesse en fonction du temps")
        ax.set_xlabel("Temps [s]")
        ax.set_ylabel("Vitesse [m/s]")

    elif afficher == "v_vs_x":
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rk45.sol(valeurs_t)[0], rk45.sol(valeurs_t)[1], color='blue', lw=2)
        ax.set_title("Graphique illustrant la vitesse en fonction de la position")
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Vitesse [m/s]")

    elif afficher == "3d":
        ax = plt.axes(projection='3d')
        ax.plot(xs=rk45.sol(valeurs_t)[0], ys=rk45.sol(valeurs_t)[1], zs=valeurs_t, color='blue', lw=2)
        ax.set_title("Graphique illustrant la vitesse et la position en fonction du temps")
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Vitesse [m/s]")
        ax.set_zlabel("Temps [s]")

    if frequence_comparative != 0:
        ax = fig.add_subplot(1, 1, 1)
        fonction = lambda time: np.sin(frequence_comparative*time)
        ax.plot(valeurs_t, fonction(valeurs_t), color='red', lw=2, label=f"sinus ayant une férquence angulaire de {frequence_comparative}rad/s")
        ax.legend()

    plt.grid()
    plt.show()
    plt.close(fig)


def afficher_point_carree_scipy(equation_reduite_ordre_1_x, equation_reduite_ordre_1_v, plage: tuple, nombre_de_points: int, conditions_initiales: tuple
                                , x=sym.symbols("x"), t=sym.symbols("t"), v=sym.symbols("v")) -> None:
    """
    Cette méthode appel la resolution_runge_kutta_ordre_2_scipy pour produire les points
    pour produire le graphique des points carrées

    Parameters
    ----------
    equation_reduite_ordre_1_x : SymPy object
        fontion associé à dx/dt ex: dx/dt = v
    equation_reduite_ordre_1_v : SymPy object
        fontion associé à dv/dt ex: dv/dt =  -w^2x + sin(t)
    plage :
        plage d'affichage pour le graphique
    nombre_de_points:
        Nombre de points désirés pour produire le graphique
    conditions_initiales:
        tuple des conditions initiales pour
        les deux variables dépendantes ex (x_0, v_0)
    x : SymPy symbol
        variable dépendante d'ordre 1
    t : SymPy symbol
        variable indépendante
    v : SymPy symbol
        variable dépendante d'ordre 2
    """
    debut, fin = plage
    x_0, v_0 = conditions_initiales
    t_0, t_f = plage

    def fusion_deux_equations(temps, x_et_v):
        valeur_x = x_et_v[0]
        valeur_v = x_et_v[1]
        fv = equation_reduite_ordre_1_v.evalf(9, {t: temps, x: valeur_x, v: valeur_v})
        fx = equation_reduite_ordre_1_x.evalf(9, {t: temps, x: valeur_x, v: valeur_v})

        return np.array([fx, fv], float)

    y_0 = np.array([x_0, v_0], float)
    roots_x = lambda temps, x_et_v: fusion_deux_equations(temps, x_et_v)[1]
    solution = scint.solve_ivp(fusion_deux_equations, plage, y_0, method="RK45", dense_output=True, events=roots_x)
    roots = solution.t_events[0]
    periode = roots[5]-roots[3]
    valeurs_t = []
    for i in range(0,nombre_de_points):
        valeurs_t.append(i*periode)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(solution.sol(valeurs_t)[0], solution.sol(valeurs_t)[1])
    ax.set_title("Graphique point carrée de la vitesse en fonction du temps")
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
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (1, 0))
    """
    # Code pour la question b)
    """
    x = sym.symbols("x")
    v = sym.symbols("v")
    equation_v = -x
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (3, 0), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (-2, 0), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (0, 2), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (0, -3), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (2, 2), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (-2, -2),  frequence_comparative=1)
    """
    # Code pour la question c)
    """
    x = sym.symbols("x")
    v = sym.symbols("v")
    equation_v = -(x**3)
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0, 30), 1001, (1, 0), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0, 15), 1001, (2, 0), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0, 25), 1001, (-1, 0), frequence_comparative=1)
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0, 10), 1001, (-3, 0), frequence_comparative=1)
    """
    # Code pour la question d)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -(x**3)
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (1, 0), "v_vs_x")
    equation_v = -x
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta(equation_x, equation_v, (0,50), 1001, (1, 0), "v_vs_x")
    """
    # Code pour la question e)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -x - (x**2 - 1)*v
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 8*np.pi), 1001, (0.5, 0), "x_vs_t")
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 8*np.pi), 1001, (0.5, 0), "v_vs_t")
    """
    # Code pour la question f)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -x - (x**2 - 1)*v
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 14*np.pi), 1001, (1, 0), "v_vs_x")
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 14*np.pi), 1001, (2, 0), "v_vs_x")
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 14*np.pi), 1001, (3, 0), "v_vs_x")
    """
    # Code pour la question g)
    """
    x = sym.symbols("x")
    t = sym.symbols("t")
    v = sym.symbols("v")
    equation_v = -x - (x**2 - 1)*v
    equation_x = v
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 8*np.pi), 1001, (1, 0), "3d")
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 8*np.pi), 1001, (2, 0), "3d")
    afficher_x_ou_v_vs_t_runge_kutta_scipy(equation_x, equation_v, (0, 8*np.pi), 1001, (3, 0), "3d")
    afficher_point_carree_scipy(equation_x, equation_v, (0, 1000), 50, (1, 0))
    """
