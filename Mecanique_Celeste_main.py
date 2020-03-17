import numpy as np

donnee_probleme: dict = {
    "labels": ["A", "B", "C"],
    "m_i": np.array([3, 4, 5]),
    "r_0": np.array([
        [1, 3],
        [-2, -1],
        [1, -1]
    ]),
}


def delta_kroneckeur(i: int, j: int) -> int:
    return int(i != j)


def Fg(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Cette méthode sert à calculer les forces gravitationnel pour n masses à certaines positions.

        $\Vec{F}_i &= -G \sum_{j \ne i}^{} m_i m_j \frac{\Vec{r}_i - \Vec{r}_j}{\abs{\Vec{r}_i - \Vec{r}_j}^3}$

    :param positions: Matrices nxk contenant les vecteurs de positions de longueur k des n masses. (np.ndarray)
    :param masses: Vecteurs de longueur k contenants les coordonnées x_i des n masses. (np.ndarray)
    :return: Vecteur de longueur n contenant les forces (scalaire) appliqué sur les n masses. (np.ndarray)
    """
    from util import constantes

    assert positions.shape[0] == masses.shape[0]
    assert len(masses.shape) == 1

    f_i: list = list()
    for i, position_i in enumerate(positions):
        masse_i = masses[i]
        position_relative = -(positions - position_i)
        constante_de_normalisation = np.linalg.norm(position_relative, axis=1)**3
        constantes_de_masse = masse_i*masses
        constante_de_poids = constantes_de_masse*(1/constante_de_normalisation)
        constante_de_poids[constante_de_poids == np.inf] = 0   # on enleve les inf pour des 0

        # on reshape la matrice avoir de pouvoir faire une multiplication element wise
        constante_de_poids = np.tile(constante_de_poids.reshape((1, 3)), (2, 1)).transpose()

        f_i.append(-constantes["G"]*np.sum(np.multiply(constante_de_poids, position_relative)))
    return np.array(f_i)


def resolution_probleme_trois_corps(masses: np.ndarray, **kwargs):
    [a, b] = kwargs.get("bornes", [0, 1])
    N = kwargs.get("resolution", 100)
    r_0 = kwargs.get("conditions_initiales", np.zeros((3, 2)))

    h = (b - a) / N
    t_points = np.arange(a, b, h)
    r_points = list()  # (dr_i/dt) = v_i -> g
    v_points = list()  # (dv_i / dt) = (1/m_i)*F_i -> f

    # initialisation des fonctions à résoudre
    f = lambda r: (1/masses)*Fg(r, masses)
    g = lambda idx: v_points[idx]

    # initialisation au premier point à t_0
    v_0 = f(r_0)
    v_points.append(v_0)
    r_points.append(r_0)

    # initialisation du deuxieme point à t_0 + h/2
    v_demi = v_0 + (h/2)*f(r_points[-1])
    r_demi = r_0 + np.tile(((h/2)*g(0)).reshape((1, 3)), (2, 1)).transpose()
    v_points.append(v_demi)
    r_points.append(r_demi)

    # résolution de l'équation différentielle avec saute mouton
    for i, t in enumerate(t_points[2:]):  # ici, l'index i est deja en retard de 2 sur l'index de t dans t_points
        v_i = v_points[i] + h*f(r_points[i+1])
        r_i = r_points[i] + np.tile((h*g(i+1)).reshape((1, 3)), (2, 1)).transpose()
        v_points.append(v_i)
        r_points.append(r_i)

    return np.array(r_points), t_points


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    R, T = resolution_probleme_trois_corps(donnee_probleme["m_i"], conditions_initiales=donnee_probleme["r_0"])
    print(R.shape)
    for i in range(3):
        ax.plot(xs=R[:, i, 0], ys=R[:, i, 1], zs=T, label=donnee_probleme["labels"][i])

    ax.set_xlabel('x [unité de distance]')
    ax.set_ylabel('y [unité de distance]')
    ax.set_zlabel('t [unité de temps]')
    plt.legend()
    plt.grid()
    plt.show()
