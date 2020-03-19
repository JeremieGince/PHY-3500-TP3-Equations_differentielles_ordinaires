import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


donnee_problemes: dict = {
    "a": {
        "labels": ["A", "B", "C"],
        "t": [0, 1],
        "m_i": np.array([3, 4, 5]),
        "r_0": np.array([
            [1, 3],
            [-2, -1],
            [1, -1]
        ]),
        "v_0": np.zeros((3, 2)),
    },
    "b": {
        "labels": ["A", "B", "C"],
        "t": [0, 10],
        "m_i": np.array([3, 4, 5]),
        "r_0": np.array([
            [1, 3],
            [-2, -1],
            [1, -1]
        ]),
        "v_0": np.zeros((3, 2)),
    },
    "c": {
        "labels": ["A", "B", "C"],
        "t": [0, 15],
        "m_i": np.array([1, 1, 1]),
        "r_0": np.array([
            [3.3030197, -0.82771837],
            [-3.3030197, 0.82771837],
            [0.0, 0.0]
        ]),
        "v_0": np.array([
            [1.587433767, 1.47221479],
            [1.587433767, 1.47221479],
            [-3.174867535, -2.94442961]
        ]),
    },
    "c1": {
        "labels": ["A", "B", "C"],
        "t": [0, 15],
        "m_i": np.array([1, 1, 1]),
        "r_0": np.array([
            [3.3030197, -0.82771837],
            [-3.3030197, 0.82771837],
            [0.0, 0.0]
        ]),
        "v_0": np.array([
            [1.587433767, 1.47221479],
            [1.587433767, 1.47221479],
            [-3.174867535, -2.94442961]
        ])+1.0,
    },
    "c2": {
        "labels": ["A", "B", "C"],
        "t": [0, 10],
        "m_i": np.array([1, 1, 1]),
        "r_0": np.array([
            [3.3030197, -0.82771837],
            [-3.3030197, 0.82771837],
            [0.0, 0.0]
        ]),
        "v_0": np.array([
            [1.587433767+1.0, 1.47221479+1.0],
            [1.587433767, 1.47221479],
            [-3.174867535, -2.94442961]
        ]),
    },
    "c3": {
        "labels": ["A", "B", "C"],
        "t": [0, 15],
        "m_i": np.array([1, 1, 1]),
        "r_0": np.array([
            [3.3030197, -0.82771837],
            [-3.3030197, 0.82771837],
            [0.0, 0.0]
        ]),
        "v_0": np.array([
            [1.587433767, 1.47221479],
            [1.587433767 + 1.0, 1.47221479 + 1.0],
            [-3.174867535, -2.94442961]
        ]),
    },
    "c4": {
        "labels": ["A", "B", "C"],
        "t": [0, 15],
        "m_i": np.array([1, 1, 1]),
        "r_0": np.array([
            [3.3030197, -0.82771837],
            [-3.3030197, 0.82771837],
            [0.0, 0.0]
        ]),
        "v_0": np.array([
            [1.587433767, 1.47221479],
            [1.587433767, 1.47221479],
            [-3.174867535 + 1.0, -2.94442961 + 1.0]
        ]),
    },
    "c5": {
        "labels": ["A", "B", "C"],
        "t": [0, 100],
        "m_i": np.array([1, 1, 1]),
        "r_0": np.array([
            [1, 3],
            [-2, -1],
            [1, -1]
        ]),
        "v_0": np.array([
            [1.587433767, 1.47221479],
            [1.587433767, 1.47221479],
            [-3.174867535, -2.94442961]
        ]),
    },
}


def delta_kroneckeur(i: int, j: int) -> int:
    return int(i != j)


def Fg(positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Cette méthode sert à calculer les forces gravitationnel pour n masses à certaines positions.

        F_i = -G sum_{i != j} ( (m_i*m_j) * (r_i - r_j)/|r_i - r_j|^3 )

    :param positions: Matrices nxk contenant les vecteurs de positions de longueur k des n masses. (np.ndarray)
    :param masses: Vecteurs de longueur k contenants les coordonnées x_i des n masses. (np.ndarray)
    :return: Matrices de vecteurs de longueur nxk contenant les forces appliqué sur les n masses dans les k directions.
             (np.ndarray)
    """
    from util import constantes

    assert positions.shape[0] == masses.shape[0]
    assert len(masses.shape) == 1

    f_i: list = list()
    for i, position_i in enumerate(positions):
        masse_i = masses[i]
        position_relative = -(positions - position_i)  # r_i - r_j
        constante_de_normalisation = np.linalg.norm(position_relative, axis=1)**3  # 1/|r_i - r_j|^3
        constantes_de_masse = masse_i*masses  # m_i*m_j
        constante_de_poids = constantes_de_masse*(1/constante_de_normalisation)  # (m_i*m_j)/|r_i - r_j|^3
        constante_de_poids[constante_de_poids == np.inf] = 0   # on remplace les inf pour des 0

        # on reshape la matrice afin de pouvoir faire une multiplication element wise
        constante_de_poids = np.tile(constante_de_poids.reshape((1, 3)), (2, 1)).transpose()

        matrice_a_summer = np.multiply(constante_de_poids, position_relative)   # (m_i*m_j) * (r_i - r_j)/|r_i - r_j|^3
        f_i.append(-constantes["G"]*np.sum(matrice_a_summer, axis=0))
    return np.array(f_i)


def resolution_probleme_trois_corps(masses: np.ndarray, **kwargs):
    [a, b] = kwargs.get("bornes", [0, 1])
    N = kwargs.get("resolution", 100)
    r_0 = kwargs.get("positions_initiales", np.zeros((3, 2)))
    v_0 = kwargs.get("vitesses_initiales", np.zeros((3, 2)))

    h = (b - a) / N
    t_points = np.arange(a, b, h)
    r_points = list()  # (dr_i/dt) = v_i -> g
    v_points = list()  # (dv_i / dt) = (1/m_i)*F_i -> f

    masses_tile = np.tile(masses.reshape((1, 3)), (2, 1)).transpose()

    # initialisation des fonctions à résoudre
    f = lambda r: (1/masses_tile)*Fg(r, masses)
    g = lambda idx: v_points[idx]

    # initialisation au premier point à t_0
    # v_0 = f(r_0)
    v_points.append(v_0)
    r_points.append(r_0)

    # initialisation du deuxieme point à t_0 + h/2
    v_demi = v_0 + (h/2)*f(r_points[-1])
    r_demi = r_0 + (h/2)*g(0)
    v_points.append(v_demi)
    r_points.append(r_demi)

    # résolution de l'équation différentielle avec saute mouton
    for i, t in enumerate(t_points[2:]):  # ici, l'index i est deja en retard de 2 sur l'index de t dans t_points
        v_i = v_points[i] + h*f(r_points[i+1])  # v_i = v_{i-2} + h*f(v_{i-1}, t_{i-1})
        r_i = r_points[i] + h*g(i+1)  # r_i = r_{i-2} + h*g(r_{i-1}, t_{i-1})
        v_points.append(v_i)
        r_points.append(r_i)

    return np.array(r_points), t_points


def simulation_affichage_3D(matrice_de_position: np.ndarray, vecteur_temps: np.ndarray,
                            titre: str = "Simulation_affichage_3D", labels=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    R, T = matrice_de_position, vecteur_temps

    for i in range(3):
        ax.plot(xs=R[:, i, 0], ys=R[:, i, 1], zs=T, label=labels[i], lw=2)

    ax.set_xlabel('x [unité de distance]', fontsize=18)
    ax.set_ylabel('y [unité de distance]', fontsize=18)
    ax.set_zlabel('t [unité de temps]', fontsize=16)
    plt.legend()
    plt.grid()
    plt.savefig(f"Simulations/{titre}.png", dpi=300)
    # plt.show()


def simulaiton_animation_2D(matrice_de_position: np.ndarray, vecteur_temps: np.ndarray,
                            titre: str = "animation_resolution_3_corps", labels=None,
                            min_frames: int = 750, echelle_temps: list = [0, 1]):
    plt.style.use('seaborn-pastel')
    fig = plt.figure()
    R, T = matrice_de_position, vecteur_temps

    xlim = (min([np.min(R[:, i, 0]) for i in range(3)]), max([np.max(R[:, i, 0]) for i in range(3)]))
    ylim = (min([np.min(R[:, i, 1]) for i in range(3)]), max([np.max(R[:, i, 1]) for i in range(3)]))

    ax = plt.axes(xlim=xlim, ylim=ylim)

    text_t = ax.text(xlim[0], ylim[1], f"t: {0}", horizontalalignment='right', verticalalignment='bottom')
    line_0, = ax.plot([], [], 'o', lw=3, label=labels[0])
    line_1, = ax.plot([], [], 'o', lw=3, label=labels[1])
    line_2, = ax.plot([], [], 'o', lw=3, label=labels[2])

    ax.set_xlabel('x [unité de distance]', fontsize=20)
    ax.set_ylabel('y [unité de distance]', fontsize=20)
    plt.legend()
    plt.grid()

    hm_frames: int = min(len(T), min_frames)

    def init():
        text_t.set_text(f"t: {0}")
        line_0.set_data([], [])
        line_1.set_data([], [])
        line_2.set_data([], [])
        return line_0, line_1, line_2, text_t,

    def animate(frame_idx):
        t_idx: int = int(frame_idx * (len(T) / hm_frames))
        text_t.set_text(f"t: {(frame_idx / hm_frames) * echelle_temps[1] :.2f} [-]")
        line_0.set_data(R[t_idx, 0, 0], R[t_idx, 0, 1])
        line_1.set_data(R[t_idx, 1, 0], R[t_idx, 1, 1])
        line_2.set_data(R[t_idx, 2, 0], R[t_idx, 2, 1])

        if t_idx > 0:
            xlim = (min([np.min(R[:t_idx, i, 0]) for i in range(3)])-1, max([np.max(R[:t_idx, i, 0]) for i in range(3)])+1)
            ylim = (min([np.min(R[:t_idx, i, 1]) for i in range(3)])-1, max([np.max(R[:t_idx, i, 1]) for i in range(3)])+1)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            text_t.set_position((xlim[0], ylim[1]))

        return line_0, line_1, line_2, text_t,

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=hm_frames, interval=20, blit=True)

    anim.save(f'Simulations/{titre}.gif', writer='imagemagick')
    # plt.show()


if __name__ == '__main__':

    for probleme, donnees in donnee_problemes.items():
        R, T = resolution_probleme_trois_corps(donnees["m_i"],
                                               positions_initiales=donnees["r_0"],
                                               vitesses_initiales=donnees["v_0"],
                                               bornes=donnees["t"], resolution=100_000)
        simulation_affichage_3D(R, T, titre=f"simulation_affichage_3D_{probleme}", labels=donnees["labels"])
        # simulaiton_animation_2D(R, T, titre=f"simulationAnimation2D-{probleme}", labels=donnees["labels"],
        #                         min_frames=1_000, echelle_temps=donnees["t"])

    # probleme = "a"
    # donnees = donnee_problemes[probleme]
    # R, T = resolution_probleme_trois_corps(donnees["m_i"],
    #                                        positions_initiales=donnees["r_0"],
    #                                        vitesses_initiales=donnees["v_0"],
    #                                        bornes=donnees["t"], resolution=100_000)
    # simulation_affichage_3D(R, T, titre=f"simulation_affichage_3D_{probleme}", labels=donnees["labels"])
    # simulaiton_animation_2D(R, T, titre=f"simulation_animation_2D_{probleme}", labels=donnees["labels"],
    #                         min_frames=1_000, echelle_temps=donnees["t"])
