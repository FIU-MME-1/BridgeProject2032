#!bin/python3

import sympy
import sympy.physics.mechanics as mech
from sympy.physics.continuum_mechanics.truss import Truss
from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos
import matplotlib.pyplot as plt
import matplotlib.colors as mcm
import labellines
import numpy as np
import os
import cv2
import copy
import pickle
from scipy.interpolate import interp1d
import pandas as pd


ff = os.path.join(os.path.dirname(__file__), 'frames')
if not os.path.isdir(ff):
    os.mkdir(ff)


class Truss2(Truss):

    def __init__(self):
        """
        Initializes the class
        """
        self._nodes = []
        self._members = {}
        self._loads = {}
        self._supports = {}
        self._node_labels = []
        self._node_positions = []
        self._node_position_x = []
        self._node_position_y = []
        self._nodes_occupied = {}
        self._reaction_loads = {}
        self._internal_forces = {}
        self._node_coordinates = {}
        self._history = {'time': [], 'max_comp': [], 'max_tens': []}
        self._sf = 3e3


    def add_node(self, label, x, y):
        """
        This method adds a node to the truss along with its name/label and its location.
        Parameters
        ==========
        label:  String or a Symbol
            The label for a node. It is the only way to identify a particular node.
        x: Sympifyable
            The x-coordinate of the position of the node.
        y: Sympifyable
            The y-coordinate of the position of the node.
        Examples
        ========
        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.nodes
        [('A', 0, 0)]
        >>> t.add_node('B', 3, 0)
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        """
        x = sympify(x)
        y = sympify(y)

        if label in self._node_labels:
            raise ValueError("Node needs to have a unique label")

        elif x in self._node_position_x and y in self._node_position_y and self._node_position_x.index(x)==self._node_position_y.index(y):
            raise ValueError("A node already exists at the given position")

        else :
            self._nodes.append((label, x, y))
            self._node_labels.append(label)
            self._node_positions.append((x, y))
            self._node_position_x.append(x)
            self._node_position_y.append(y)
            self._node_coordinates[label] = [x, y]

    def remove_node(self, label):
        """
        This method removes a node from the truss.
        Parameters
        ==========
        label:  String or Symbol
            The label of the node to be removed.
        Examples
        ========
        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.nodes
        [('A', 0, 0)]
        >>> t.add_node('B', 3, 0)
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        >>> t.remove_node('A')
        >>> t.nodes
        [('B', 3, 0)]
        """
        for i in range(len(self.nodes)):
            if self._node_labels[i] == label:
                x = self._node_position_x[i]
                y = self._node_position_y[i]

        if label not in self._node_labels:
            raise ValueError("No such node exists in the truss")

        else:
            members_duplicate = self._members.copy()
            for member in members_duplicate:
                if label == self._members[member][0] or label == self._members[member][1]:
                    raise ValueError("The node given has members already attached to it")
            self._nodes.remove((label, x, y))
            self._node_labels.remove(label)
            self._node_positions.remove((x, y))
            self._node_position_x.remove(x)
            self._node_position_y.remove(y)
            if label in list(self._loads):
                self._loads.pop(label)
            if label in list(self._supports):
                self._supports.pop(label)
            self._node_coordinates.pop(label)

    def change_node_label(self, label, new_label):
        """
        This method changes the label of a node.
        Parameters
        ==========
        label: String or Symbol
            The label of the node for which the label has
            to be changed.
        new_label: String or Symbol
            The new label of the node.
        Examples
        ========
        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        >>> t.change_node_label('A', 'C')
        >>> t.nodes
        [('C', 0, 0), ('B', 3, 0)]
        """
        if label not in self._node_labels:
            raise ValueError("No such node exists for the Truss")
        elif new_label in self._node_labels:
            raise ValueError("A node with the given label already exists")
        else:
            for node in self._nodes:
                if node[0] == label:
                    self._nodes[self._nodes.index((label, node[1], node[2]))] = (new_label, node[1], node[2])
                    self._node_labels[self._node_labels.index(node[0])] = new_label
                    self._node_coordinates[new_label] = self._node_coordinates[label]
                    self._node_coordinates.pop(label)
                    if node[0] in list(self._supports):
                        self._supports[new_label] = self._supports[node[0]]
                        self._supports.pop(node[0])
                    if new_label in list(self._supports):
                        if self._supports[new_label] == 'pinned':
                            if 'R_'+str(label)+'_x' in list(self._reaction_loads) and 'R_'+str(label)+'_y' in list(self._reaction_loads):
                                self._reaction_loads['R_'+str(new_label)+'_x'] = self._reaction_loads['R_'+str(label)+'_x']
                                self._reaction_loads['R_'+str(new_label)+'_y'] = self._reaction_loads['R_'+str(label)+'_y']
                                self._reaction_loads.pop('R_'+str(label)+'_x')
                                self._reaction_loads.pop('R_'+str(label)+'_y')
                            self._loads[new_label] = self._loads[label]
                            for load in self._loads[new_label]:
                                if load[1] == 90:
                                    load[0] -= Symbol('R_'+str(label)+'_y')
                                    if load[0] == 0:
                                        self._loads[label].remove(load)
                                    break
                            for load in self._loads[new_label]:
                                if load[1] == 0:
                                    load[0] -= Symbol('R_'+str(label)+'_x')
                                    if load[0] == 0:
                                        self._loads[label].remove(load)
                                    break
                            self.apply_load(new_label, Symbol('R_'+str(new_label)+'_x'), 0)
                            self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                            self._loads.pop(label)
                        elif self._supports[new_label] == 'roller':
                            self._loads[new_label] = self._loads[label]
                            for load in self._loads[label]:
                                if load[1] == 90:
                                    load[0] -= Symbol('R_'+str(label)+'_y')
                                    if load[0] == 0:
                                        self._loads[label].remove(load)
                                    break
                            self.apply_load(new_label, Symbol('R_'+str(new_label)+'_y'), 90)
                            self._loads.pop(label)
                    else:
                        if label in list(self._loads):
                            self._loads[new_label] = self._loads[label]
                            self._loads.pop(label)
                    for member in list(self._members):
                        if self._members[member][0] == node[0]:
                            self._members[member][0] = new_label
                            self._nodes_occupied[(new_label, self._members[member][1])] = True
                            self._nodes_occupied[(self._members[member][1], new_label)] = True
                            self._nodes_occupied.pop(tuple([label, self._members[member][1]]))
                            self._nodes_occupied.pop(tuple([self._members[member][1], label]))
                        elif self._members[member][1] == node[0]:
                            self._members[member][1] = new_label
                            self._nodes_occupied[(self._members[member][0], new_label)] = True
                            self._nodes_occupied[(new_label, self._members[member][0])] = True
                            self._nodes_occupied.pop(tuple([self._members[member][0], label]))
                            self._nodes_occupied.pop(tuple([label, self._members[member][0]]))

    def solve(self):
        """
        This method solves for all reaction forces of all supports and all internal forces
        of all the members in the truss, provided the Truss is solvable.
        A Truss is solvable if the following condition is met,
        2n >= r + m
        Where n is the number of nodes, r is the number of reaction forces, where each pinned
        support has 2 reaction forces and each roller has 1, and m is the number of members.
        The given condition is derived from the fact that a system of equations is solvable
        only when the number of variables is lesser than or equal to the number of equations.
        Equilibrium Equations in x and y directions give two equations per node giving 2n number
        equations. However, the truss needs to be stable as well and may be unstable if 2n > r + m.
        The number of variables is simply the sum of the number of reaction forces and member
        forces.
        .. note::
           The sign convention for the internal forces present in a member revolves around whether each
           force is compressive or tensile. While forming equations for each node, internal force due
           to a member on the node is assumed to be away from the node i.e. each force is assumed to
           be compressive by default. Hence, a positive value for an internal force implies the
           presence of compressive force in the member and a negative value implies a tensile force.
        Examples
        ========
        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node("node_1", 0, 0)
        >>> t.add_node("node_2", 6, 0)
        >>> t.add_node("node_3", 2, 2)
        >>> t.add_node("node_4", 2, 0)
        >>> t.add_member("member_1", "node_1", "node_4")
        >>> t.add_member("member_2", "node_2", "node_4")
        >>> t.add_member("member_3", "node_1", "node_3")
        >>> t.add_member("member_4", "node_2", "node_3")
        >>> t.add_member("member_5", "node_3", "node_4")
        >>> t.apply_load("node_4", magnitude=10, direction=270)
        >>> t.apply_support("node_1", type="pinned")
        >>> t.apply_support("node_2", type="roller")
        >>> t.solve()
        >>> t.reaction_loads
        {'R_node_1_x': 0, 'R_node_1_y': 20/3, 'R_node_2_y': 10/3}
        >>> t.internal_forces
        {'member_1': 20/3, 'member_2': 20/3, 'member_3': -20*sqrt(2)/3, 'member_4': -10*sqrt(5)/3, 'member_5': 10}
        """
        count_reaction_loads = 0
        for node in self._nodes:
            if node[0] in list(self._supports):
                if self._supports[node[0]]=='pinned':
                    count_reaction_loads += 2
                elif self._supports[node[0]]=='roller':
                    count_reaction_loads += 1
        if 2*len(self._nodes) != len(self._members) + count_reaction_loads:
            raise ValueError("The given truss cannot be solved")
        coefficients_matrix = [[0 for i in range(2*len(self._nodes))] for j in range(2*len(self._nodes))]
        load_matrix = zeros(2*len(self.nodes), 1)
        load_matrix_row = 0
        for node in self._nodes:
            if node[0] in list(self._loads):
                for load in self._loads[node[0]]:
                    if load[0] != Symbol('R_'+str(node[0])+'_x') and load[0] != Symbol('R_'+str(node[0])+'_y'):
                        load_matrix[load_matrix_row] -= load[0]*cos(pi*load[1]/180)
                        load_matrix[load_matrix_row + 1] -= load[0]*sin(pi*load[1]/180)
            load_matrix_row += 2
        cols = 0
        row = 0
        for node in self._nodes:
            if node[0] in list(self._supports):
                if self._supports[node[0]]=='pinned':
                    coefficients_matrix[row][cols] += 1
                    coefficients_matrix[row+1][cols+1] += 1
                    cols += 2
                elif self._supports[node[0]]=='roller':
                    coefficients_matrix[row+1][cols] += 1
                    cols += 1
            row += 2
        for member in list(self._members):
            start = self._members[member][0]
            end = self._members[member][1]
            length = sqrt((self._node_coordinates[start][0]-self._node_coordinates[end][0])**2 + (self._node_coordinates[start][1]-self._node_coordinates[end][1])**2)
            start_index = self._node_labels.index(start)
            end_index = self._node_labels.index(end)
            horizontal_component_start = (self._node_coordinates[end][0]-self._node_coordinates[start][0])/length
            vertical_component_start = (self._node_coordinates[end][1]-self._node_coordinates[start][1])/length
            horizontal_component_end = (self._node_coordinates[start][0]-self._node_coordinates[end][0])/length
            vertical_component_end = (self._node_coordinates[start][1]-self._node_coordinates[end][1])/length
            coefficients_matrix[start_index*2][cols] += horizontal_component_start
            coefficients_matrix[start_index*2+1][cols] += vertical_component_start
            coefficients_matrix[end_index*2][cols] += horizontal_component_end
            coefficients_matrix[end_index*2+1][cols] += vertical_component_end
            cols += 1
        forces_matrix = (Matrix(coefficients_matrix)**-1)*load_matrix
        self._reaction_loads = {}
        i = 0
        min_load = inf
        for node in self._nodes:
            if node[0] in list(self._loads):
                for load in self._loads[node[0]]:
                    if type(load[0]) not in [Symbol, Mul, Add]:
                        min_load = min(min_load, load[0])
        for j in range(len(forces_matrix)):
            if type(forces_matrix[j]) not in [Symbol, Mul, Add]:
                if abs(forces_matrix[j]/min_load) <1E-10:
                    forces_matrix[j] = 0
        for node in self._nodes:
            if node[0] in list(self._supports):
                if self._supports[node[0]]=='pinned':
                    self._reaction_loads['R_'+str(node[0])+'_x'] = forces_matrix[i]
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i+1]
                    i += 2
                elif self._supports[node[0]]=='roller':
                    self._reaction_loads['R_'+str(node[0])+'_y'] = forces_matrix[i]
                    i += 1
        for member in list(self._members):
            self._internal_forces[member] = forces_matrix[i]
            i += 1
        return

    def get_member_types(self):
        self.member_types = {i: 'road' if i.find('road') > -1 else 'beam' for i in self.members}

    def get_member_lengths(self):
        def rel_f(l):
            if l < 1.847:
                return 1. - 0.14633 * l**2
            else:
                return 1.7094 / l**2
        self.member_lengths = {i: ((self._node_position_x[self._node_labels.index(self.members[i][0])] -
                                    self._node_position_x[self._node_labels.index(self.members[i][1])]) ** 2 +
                                   (self._node_position_y[self._node_labels.index(self.members[i][0])] -
                                    self._node_position_y[self._node_labels.index(self.members[i][1])]) ** 2
                                   ) ** 0.5 for i in self.members}
        self.compressive_weaknesses = {i[0]: rel_f(i[1]) for i in self.member_lengths.items()}
        # wood: E = 6.9 GPa, density = 0.7 g/cc, sigy = 34.5 MPa
        # calculate critical length
        # kg m s N Pa J
        # calculate critical load(length)
        # mass / m (if aluminum, 5cm by 5cm square rod, = 6.75 kg / m ??? (5 cm ~ 2 in
        # density = 2.7 g / cm**3  (
        #  A = 25cm**2 (2.5e-3 m**2)
        # I = 52.08 cm**4 = 5.21e-7 m**4
        # R = (I/A)**0.5 = 1.44 cm 1.44e-2 m
        # E = 69 GPa (6.9e10 Pa)
        # Sigma_yield = 83 MPa (8.3e7 Pa)
        # Scrit = 128
        # L crit = 1.84704 m
        # Tensile Failure: 8.3e7*2.5e-3 = 207.5 kN
        # Johnson's Formula (short columns): F = A*sig* [1-(sig/E)*(L/(2*pi*R))**2]
        # F = 207.5 kN * [ 1 - 0.14633 * L**2]
        # Euler's Formula (above L = 1.847 m) F = pi**2 * E * I/ L**2
        # F = 6.9e10 * pi**2 * 5.208e-7 /L**2
        # F = 1.7094 * 207.5 kN / L**2
        # self.member_buckling_loads = {i[0]: i[1]}

    def get_member_masses(self, masses):
        self.member_masses = {i: self.member_lengths[i] * masses[self.member_types[i]] for i in self.members}

    def get_node_masses(self, masses):
        self.node_masses = {i[0]: masses['node'] for i in self.nodes}

    def update_history(self, time):
        self._history['time'].append(time)
        mt = max(self._internal_forces.values())
        mc = min([self._internal_forces[i]/self.compressive_weaknesses[i] for i in self._internal_forces])
        self._history['max_comp'].append(-mc)
        self._history['max_tens'].append(mt)

    def plotter(self, time=0, name='', group=''):
        scale = 10000
        cdict = {'red': [(0.0, 0.0, 0.0),
                         (0.5, 0.0, 0.0),
                         (1.0, 1.0, 1.0)],

                 'green': [(0.0, 0.0, 0.0),
                           (1.0, 0.0, 0.0)],

                 'blue': [(0.0, 1.0, 1.0),
                          (0.5, 0.0, 0.0),
                          (1.0, 0.0, 0.0)]}
        bx = [[float(self._node_position_x[self._node_labels.index(self.members[i][0])]),
               float(self._node_position_x[self._node_labels.index(self.members[i][1])])]
              for i in self.members]
        rbx = [[float(self._node_position_x[self._node_labels.index(self.members[i][0])]),
               float(self._node_position_x[self._node_labels.index(self.members[i][1])])]
              for i in self.members if i.find('road') > -1]
        xmin = np.min(rbx)
        xmax = np.max(rbx)
        by = [[float(self._node_position_y[self._node_labels.index(self.members[i][0])]),
               float(self._node_position_y[self._node_labels.index(self.members[i][1])])]
              for i in self.members]
        rx = [float(i[1]) for i in self.nodes if i[0].find('road') > -1]
        ry = [float(i[2]) for i in self.nodes if i[0].find('road') > -1]
        ymin, ymax = [f(by) for f in [np.min, np.max]]
        dn = mcm.TwoSlopeNorm(vcenter=0, vmin=-1*self._sf,
                              vmax=self._sf)
        cmap = mcm.LinearSegmentedColormap('stress', cdict)

        colors = [cmap(dn(float(i[1]))) if float(i[1]) > 0 else
                  cmap(dn(float(i[1])/float(self.compressive_weaknesses[i[0]])))
                  for i in self.internal_forces.items()]
        labs = ['%.0f' % self.internal_forces[i] for i in self.members]
        fig, ax = plt.subplots(2, 1, figsize=[12, 12])
        plt.sca(ax[0])
        plt.plot(self._history['time'], self._history['max_comp'], 'b-', label='Compression')
        plt.plot(self._history['time'], self._history['max_tens'], 'r-', label='Tension')
        plt.plot([0, 1], [self._sf, self._sf], 'k:', label='Load Limit')
        plt.xlim((0, 1))
        plt.ylim((0, 2e4))
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        maxf = max([max(self._history['max_comp']), max(self._history['max_tens'])]) / self._sf
        textstr = "Maximum Stress Reached %.4f" % maxf + r' $\times$ design limit'
        # place a text box in upper left in axes coords
        ax[0].text(0.05, 0.95, textstr, transform=ax[0].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        if time > 0:
            labellines.labelLines(plt.gca().get_lines(), xvals=[time*i for i in [0.2, 0.6, 0.5/time]])
        plt.sca(ax[1])
        [plt.plot(x, y, '-', c=c, label=l) for x, y, c, l in zip(bx, by, colors, labs)]
        massx = time*xmax
        f = interp1d(rx, ry)
        massy = f(massx)+0.2*ymax
        labellines.labelLines(plt.gca().get_lines(), xvals=[np.mean(bi) for bi in bx])
        plt.scatter(massx, massy, marker=r'$\Phi$', c='k', s=70)
        plt.plot(np.array(bx).T, np.array(by).T, 'ok')

        for i in self.reaction_loads:
            if self.reaction_loads[i] == 0:
                continue
            ni = self._node_labels.index(i[2:-2])
            x, y = self._node_positions[ni]
            dir = np.sign(self.reaction_loads[i])
            if i[-2:].find('_x'):
                # horiz
                dx = dir * 0.2
                dy = 0
            elif i[-2:].find('_y'):
                # vert
                dx = 0
                dy = dir * 0.2
            w = self.reaction_loads[i]/scale
            plt.arrow(x, y, dy, dx, width=w,
                      head_width=w*1.5, head_length=w*0.5)
        for i in self.loads:
            for ii in self.loads[i]:
                mags = []
                dirs = []
                if isinstance(ii[0], sympy.core.numbers.Float):
                    mags.append(ii[0])
                    dirs.append(ii[1])
            dx = [0.2 * np.sin(np.radians(float(d))) for d in dirs]
            dy = [0.2 * np.cos(np.radians(float(d))) for d in dirs]
            ww = np.divide(mags, sum(mags))
            dx = np.multiply(dx, ww).sum()
            dy = np.multiply(dy, ww).sum()

            x, y = self._node_positions[self._node_labels.index(i)]
            w = float(np.sum(mags)/scale)
            plt.arrow(float(x), float(y), float(dy), float(dx), width=w,
                          head_width=w * 1.5, head_length=w * 0.5, fc='green')
        if (ymax-ymin)*2 > xmax - xmin:
            buf = (ymax - ymin)*2 - (xmax - xmin)
        else: buf = 0.5
        plt.xlim((float(xmin)-buf, float(xmax)+buf))
        plt.ylim((float(ymin)-buf/2, float(ymin) + float(xmax)/2 + buf/2))
        plt.title(name + ' ' + group)
        plt.savefig(os.path.join(ff, '%04i.png' % (time*200)), dpi=120)
        plt.close()


def make_bridge(length=8, segments=4, height=3):
    tt = Truss2()
    # build road
    for i in range(segments + 1):
        tt.add_node('road_joint_%i' % i, i * length/segments, 0)
    for i in range(1, segments + 1):
        tt.add_node('truss_node_%i' % i, (i - 0.5) * length/segments, height)
        tt.add_member('beam_%ia' % i, 'road_joint_%i' % (i-1), 'truss_node_%i' % i)
        tt.add_member('beam_%ib' % i, 'road_joint_%i' % i, 'truss_node_%i' % i)
    for i in range(2, segments + 1):
        tt.add_member('beam_%ic' % i, 'truss_node_%i' % (i - 1), 'truss_node_%i' % i)
    for i in range(1, segments + 1):
        tt.add_member('road_section_%i' % i, 'road_joint_%i' % (i-1), 'road_joint_%i' % i)
    tt.apply_support('road_joint_0', 'pinned')
    tt.apply_support('road_joint_%i' % segments, 'roller')
    return tt


def make_f_bridge(length=8, segments=4, f_y_road=0, f_y_pins=3, f_x_pins=0):
    """This is one approach to defining node positions by function.

    This function will always place one less horizontal support beam than road segments.
    You can make much better bridges if you discover other valid truss designs...
    Parameters:
        length: float or int
        segments: int
        f_y_road: int, float, or Callable
        f_y_pins: int, float, or Callable
        f_x_pins: int, float, or Callable

        Any Callable passed to the parameters above needs to have the signature:
         f(x, xmax) -> float or int
         where x is the current x position along the bridge (assuming equal spacing)
         and xmax is the length of the bridge."""
    tt = Truss2()
    if isinstance(f_y_road, (int, float)):
        val = f_y_road

        def f_y_road(x, xmax):
            return val

    if isinstance(f_y_pins, (int, float)):
        val = f_y_pins

        def f_y_pins(x, xmax):
            return val

    if isinstance(f_x_pins, (int, float)):
        val = f_x_pins
        def f_x_pins(x, xmax):
            return val + x

    # build road
    for i in range(segments + 1):
        tt.add_node('road_joint_%i' % i, i * length/segments, f_y_road(i * length/segments, length))
    for i in range(1, segments + 1):
        xp = f_x_pins(i * length/segments, length)
        tt.add_node('truss_node_%i' % i, xp, f_y_pins(xp, length))
        tt.add_member('beam_%ia' % i, 'road_joint_%i' % (i-1), 'truss_node_%i' % i)
        tt.add_member('beam_%ib' % i, 'road_joint_%i' % i, 'truss_node_%i' % i)
    for i in range(2, segments + 1):
        tt.add_member('beam_%ic' % i, 'truss_node_%i' % (i - 1), 'truss_node_%i' % i)
    for i in range(1, segments + 1):
        tt.add_member('road_section_%i' % i, 'road_joint_%i' % (i-1), 'road_joint_%i' % i)
    tt.apply_support('road_joint_0', 'pinned')
    tt.apply_support('road_joint_%i' % segments, 'roller')
    return tt


def apply_self_load(truss, masses):
    g = 9.81 #(m/s**2)
    truss.get_member_lengths()
    truss.get_member_types()
    truss.get_member_masses(masses)
    truss.get_node_masses(masses)
    efm = {i[0]: truss.node_masses[i[0]] for i in truss.nodes}
    for member in truss.members:
        a, b = truss.members[member]
        efm[a] += truss.member_masses[member] / 2
        efm[b] += truss.member_masses[member] / 2
    for node in truss._node_labels:
        truss.apply_load(node, magnitude=efm[node]*g, direction=270)
    return


def lever_rule(fun, road, pos, v):
    g = 9.81
    keylist = list(road.keys())
    dist = {i: abs(road[i][1]-pos) for i in keylist}

    dl2 = [dist[i] for i in dist]
    dl2.sort()
    lim = max(dl2[0:2])
    nodes = {i: dist[i] for i in dist if dist[i] <= lim}
    nl = [(i[0], i[1]) for i in nodes.items()]
    ws = {nl[i][0]: nl[-(i+1)][1]/sum(dl2[0:2]) for i in range(len(nodes))}
    # w0 = d1/(d0+d1)
    # w1 = d0/(d0+d1)
    load = g * v
    for i in ws:
        fun(road[keylist[i]][0], load * ws[i], 270)
    return


def apply_moving_loads(truss, t, v, tprev=None):
    road_pins = [i for i in truss._node_labels if i.find('road') > -1]
    road_pin_numbers = [int(i.split('_')[-1]) for i in road_pins]
    road_nodes = {i[0]: truss.nodes[truss._node_labels.index(i[1])] for i in zip(road_pin_numbers, road_pins)}
    road_xvals = [i[1] for i in road_nodes.values()]
    load_pos = t * (max(road_xvals) - min(road_xvals))
    if tprev is None:
        pass
    else:
        prev_pos = tprev * (max(road_xvals) - min(road_xvals))
        lever_rule(truss.remove_load, road_nodes, prev_pos, v)
    lever_rule(truss.apply_load, road_nodes, load_pos, v)
    return


def save_truss(tt, name=None, group=None):
    if name is None:
        name = 'truss' + input("What # round is this? ")
    if group is None:
        group = input("What is your group name? ")
    h = open(name + group + '.pickle', 'wb')
    pickle.dump(tt, h)
    h.close()


def smart_pfind(path):
    ll = os.listdir(path)
    ll2 = [li for li in ll if li.find('.pickle') > -1]
    return ll2[int(input('Pick by number:\n' +
                         '\n'.join(['%i: %s' % (i, li) for i, li in enumerate(ll2)])+'\n'))]


def get_truss(name=None):
    if name is None:
        name = smart_pfind('.')
    h = open(name, 'rb')
    tt = pickle.load(h)
    h.close()
    return tt


def truss_simulator(truss, moving_mass=50, fixed_loads=None, ligament_masses=None, tsteps=None, fps=2, show=True, name=None, group=None):
    """Different Rounds may have different lengths, number of segments, or arbitrary requirements on shape (truss below road,
    truss above road, """
    if name is None:
        name = 'truss' + input("What # round is this? ")
    if group is None:
        group = input("What is your group name? ")
    if fixed_loads is None:
        fixed_loads = {}
    if ligament_masses is None:
        ligament_masses = {'road': 20,  # kg / m
                           'beam': 4,  # kg / m
                           'node': 2}  # kg
    if tsteps is None:
        rnodes = len([i for i in truss._node_labels if i.find('road') > -1])
        tsteps = np.linspace(0, 1, 2*rnodes-1)
    apply_self_load(truss, ligament_masses)
    apply_moving_loads(truss, tsteps[0], moving_mass)
    truss.solve()
    truss.update_history(tsteps[0])
    truss.plotter(tsteps[0], name=name, group=group)
    for ii, t in enumerate(tsteps[1:]):
        oldloads = copy.deepcopy(truss.loads)
        apply_moving_loads(truss, t, moving_mass, tsteps[ii])
        changes = ['%s: %s -> %s' % (i, oldloads[i][-1], truss.loads[i][-1]) for i in truss.loads if oldloads[i] != truss.loads[i]]
        print(ii + 1, t, tsteps[ii])
        print(*changes, sep='\n')
        truss.solve()
        truss.update_history(t)
        truss.plotter(t, name=name, group=group)
    if show:
        make_video(fps, )
        if input('Save this result? (y/N) ').lower() in ['y', 'yes', 'ye']:
            save_truss(truss, name, group)


def make_video(fps, vn='video.mp4'):
    # image_folder = 'frames'
    video_name = vn
    images = [img for img in os.listdir(ff) if img.endswith(".png")]
    images.sort()
    frame = cv2.imread(os.path.join(ff, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(ff, image)))

    cv2.destroyAllWindows()
    video.release()

