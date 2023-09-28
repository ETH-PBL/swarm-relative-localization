#!/usr/bin/env python3

#  
#   Copyright (C) 2023 ETH Zurich
#   All rights reserved.
#  
#   This software may be modified and distributed under the terms
#   of the GPL-3.0 license.  See the LICENSE file for details.
#  
#   Author: Dominik Schindler
#  

import numpy as np


class Problem:
    def __init__(self, d_ij_init):
        # Current distance measurement.
        self.d = d_ij_init
        # Time constant of exponential decay for residual weighting.
        self.tau = 15.0;

        # Quadratic, Linear, and constant term of the resulting QP.
        # The terms are split into components that are quadratic, linear, and
        # constant in the unknown transformation matrix.
        self.Q11C = np.zeros([3, 3])
        self.Q11L = np.zeros([3, 3])
        self.Q11M = np.zeros([3, 3])

        self.Q12C = np.zeros([3, 1])
        self.Q12L = np.zeros([3, 1])

        self.Q22C = np.ones([1, 1])

        self.c1C = np.zeros([3, 1])
        self.c1L = np.zeros([3, 3])
        self.c1M = np.zeros([3, 3])

        self.c2C = np.zeros([1, 1])
        self.c2L = np.zeros([1, 3])

        self.eC = np.zeros([1, 1])
        self.eL = np.zeros([1, 3])
        self.eM = np.zeros([3, 3])

    def add_data(self, v_i, v_j, d_ij_next, dt):
        d_ij = self.d
        self.d = d_ij_next

        mu = np.array([v_i]).T * dt
        nu = -np.array([v_j]).T * dt

        gamma = (d_ij**2 - d_ij_next**2 - np.linalg.norm(mu)**2 - np.linalg.norm(nu)**2)/(2*d_ij)
        delta = (-1/d_ij) * h(mu.T, nu).T

        decay = np.exp(-dt/self.tau)

        self.eC = decay * (self.eC
            + (mu.T@self.Q11C@mu + 2*nu.T@self.Q11L@mu +nu.T@self.Q11M@nu)
            + 2*gamma*(mu.T@self.Q12C + nu.T@self.Q12L)
            + (self.Q22C*gamma**2)
            - 2*mu.T@self.c1C
            - 2*gamma*self.c2C)

        self.eL = decay * (self.eL
            + (h(mu.T@(self.Q11C+self.Q11C.T), nu) + h(mu.T, (self.Q11M+self.Q11M.T)@nu))
            + 2*(h(mu.T, self.Q11L@mu) + h(nu.T@self.Q11L, nu))
            + 2*(h(gamma*mu.T, self.Q12L) + h(gamma*self.Q12C.T, nu) + (mu.T@self.Q12C + nu.T@self.Q12L)@delta.T)
            + 2*gamma*self.Q22C*delta.T
            - 2*(mu.T@self.c1L + h(self.c1C.T, nu) + nu.T@self.c1M)
            - 2*(gamma*self.c2L + self.c2C@delta.T))

        self.eM = decay * (self.eM
            + (h_for_transpose(h_for_transpose(self.Q11M.T, mu).T, mu) + 2*h(h_for_transpose(self.Q11L.T, mu).T, nu) + h(h(self.Q11C.T, nu).T, nu))
            + 2*delta@(h(mu.T, self.Q12L) + h(self.Q12C.T, nu))
            + self.Q22C*delta@delta.T
            - 2*(h_for_transpose(self.c1M.T, mu) + h(self.c1L.T, nu))
            - 2*(delta@self.c2L))

        self.c1C = decay * (self.c1C
            - (1/d_ij) * mu@self.c2C
            - (self.Q11C@mu + self.Q11L.T@nu)
            + (1/d_ij) * mu@(self.Q12C.T@mu + self.Q12L.T@nu)
            - self.Q12C*gamma
            + (self.Q22C*gamma/d_ij) * mu)

        self.c1L = decay * (self.c1L
            - (1/d_ij) * (mu@self.c2L + h(np.eye(3), nu*self.c2C))
            - (h(self.Q11C, nu) + h(np.eye(3), self.Q11L@mu) + h_for_transpose(self.Q11L.T, mu) + h(np.eye(3), self.Q11M@nu))
            + (1/d_ij) * (h(np.eye(3), nu@(self.Q12C.T@mu+self.Q12L.T@nu)) + mu@(h(mu.T, self.Q12L)+h(self.Q12C.T, nu)))
            - (h(np.eye(3), self.Q12L*gamma) + self.Q12C@delta.T)
            + (self.Q22C/d_ij) * (mu@delta.T + h(np.eye(3), gamma*nu)))

        self.c1M = decay * (self.c1M
            - (1/d_ij) * nu@self.c2L
            - (h(self.Q11L, nu) + h_for_transpose(self.Q11M, mu))
            + (1/d_ij) * nu@(h(mu.T, self.Q12L) + h(self.Q12C.T, nu))
            - self.Q12L@delta.T
            + (self.Q22C/d_ij) * nu@delta.T)

        self.c2C = (decay*d_ij_next/d_ij) * (self.c2C - self.Q12C.T@mu - self.Q12L.T@nu - self.Q22C*gamma)
        self.c2L = (decay*d_ij_next/d_ij) * (self.c2L - h(self.Q12C.T, nu) - h(mu.T, self.Q12L) - self.Q22C*delta.T)

        self.Q11C = decay * (self.Q11C + (1/d_ij) * (-self.Q12C@mu.T - mu@self.Q12C.T + (self.Q22C/d_ij)*mu@mu.T))
        self.Q11L = decay * (self.Q11L + (1/d_ij) * (-self.Q12L@mu.T - nu@self.Q12C.T + (self.Q22C/d_ij)*nu@mu.T))
        self.Q11M = decay * (self.Q11M + (1/d_ij) * (-self.Q12L@nu.T - nu@self.Q12L.T + (self.Q22C/d_ij)*nu@nu.T))

        self.Q12C = (decay * d_ij_next/d_ij) * (self.Q12C - (self.Q22C/d_ij)*mu)
        self.Q12L = (decay * d_ij_next/d_ij) * (self.Q12L - (self.Q22C/d_ij)*nu)

        self.Q22C = 1 + decay * (d_ij_next/d_ij)**2 * self.Q22C

    def solve(self):
        minimum_value = None
        minimizer_angle = None
        qp_minimizer = None

        for i in range(0, 359):
            angle = np.pi * i / 180.0
            Q = self.compute_quadratic_coeff(angle)
            c = self.compute_linear_coeff(angle)
            e = self.compute_constant_coeff(angle)
            d = self.d

            det, detx, dety, detz, detw = self.compute_qp_determinants(Q, c, d)
            poly = detx**2 + dety**2 + detz**2 - (d**2)*det**2 + 2*d*det*detw - detw**2

            for r in [r.real for r in poly.roots() if  abs(r.real) > 1e3 * abs(r.imag)]:
                div = det(r)

                x = detx(r)/div
                y = dety(r)/div
                z = detz(r)/div
                w = detw(r)/div

                sol = np.array([[x], [y], [z], [w]])

                v = sol.transpose() @ Q @ sol - 2*sol.transpose() @ c + e
                if minimum_value is None or v < minimum_value:
                    minimum_value = v.item()
                    qp_minimizer = sol
                    minimizer_angle = angle

        return qp_minimizer[:3, 0], minimizer_angle, minimum_value

    def compute_quadratic_coeff(self, angle):
        s = np.sin(angle)
        c = np.cos(angle)
        T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        Q11 = self.Q11C + T @ self.Q11L + self.Q11L.T @ T.T + T @ self.Q11M @ T.T
        Q12 = self.Q12C + T @ self.Q12L
        Q22 = self.Q22C
        return np.block([[Q11, Q12], [Q12.T, Q22]])

    def compute_linear_coeff(self, angle):
        s = np.sin(angle)
        c = np.cos(angle)
        T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        v = np.ones([3, 1])

        c1 = self.c1C + self.c1L @ T @ v + T @ self.c1M @ T @ v
        c2 = self.c2C + self.c2L @ T @ v

        return np.block([[c1], [c2]])

    def compute_constant_coeff(self, angle):
        s = np.sin(angle)
        c = np.cos(angle)
        T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        v = np.ones([3, 1])
        return self.eC + self.eL @ T @ v + v.T @ T.T @ self.eM @ T @ v

    def compute_qp_determinants(self, Q, c, d):
        A = [[Q[i, j] for j in range(4)] for i in range(4)]
        b = [c[i, 0] for i in range(4)]

        A[0][0] = np.polynomial.Polynomial([Q[0, 0], -1])
        A[1][1] = np.polynomial.Polynomial([Q[1, 1], -1])
        A[2][2] = np.polynomial.Polynomial([Q[2, 2], -1])
        A[3][3] = np.polynomial.Polynomial([Q[3, 3], 1])
        b[3] = np.polynomial.Polynomial([c[3, 0], d])

        det = det4x4(A[0], A[1], A[2], A[3])
        detx = det4x4(b, A[1], A[2], A[3])
        dety = det4x4(A[0], b, A[2], A[3])
        detz = det4x4(A[0], A[1], b, A[3])
        detw = det4x4(A[0], A[1], A[2], b)

        return det, detx, dety, detz, detw


# The arguments a1, a2, a3 are the columns (or rows) of the matrix.
def det3x3(a1, a2, a3):
    return ( a1[0] * a2[1] * a3[2]
           + a1[1] * a2[2] * a3[0]
           + a1[2] * a2[0] * a3[1]
           - a3[0] * a2[1] * a1[2]
           - a3[1] * a2[2] * a1[0]
           - a3[2] * a2[0] * a1[1])

def det4x4(a1, a2, a3, a4):
    return (  a1[0] * det3x3(a2[1:], a3[1:], a4[1:])
            - a2[0] * det3x3(a1[1:], a3[1:], a4[1:])
            + a3[0] * det3x3(a1[1:], a2[1:], a4[1:])
            - a4[0] * det3x3(a1[1:], a2[1:], a3[1:]))

def h(u, v):
    num_rows, _ = u.shape
    result = np.zeros(u.shape)

    for i in range(num_rows):
        ux, uy, uz = u[i, 0], u[i, 1], u[i, 2]
        vx, vy, vz = v[0, 0], v[1, 0], v[2, 0]

        a = 0.5 * (ux*vx + uy*vy)
        b = 0.5 * (ux*vy - uy*vx)

        result[i, :] = np.array([(a+b), (a-b), uz*vz])

    return result

def h_for_transpose(M, v):
    assert M.shape == (3, 3)
    return np.block([[h(v.T, M[0, None, :].T)],
                     [h(v.T, M[1, None, :].T)],
                     [h(v.T, M[2, None, :].T)]])



if __name__ == '__main__':
    # Heading angles of the agent i and j's odometry frame w.r.t. the world frame.
    heading_i = 77 * np.pi/180.0
    heading_j = 166 * np.pi/180.0

    T_WLi = np.array([[np.cos(heading_i), -np.sin(heading_i), 0],
                      [np.sin(heading_i),  np.cos(heading_i), 0],
                      [0,                  0,                 1]])

    T_WLj = np.array([[np.cos(heading_j), -np.sin(heading_j), 0],
                      [np.sin(heading_j),  np.cos(heading_j), 0],
                      [0,                  0,                 1]])

    # Positions of agent i and j in the world coordinate frame.
    W_p_i = np.array([0, 0, 0])
    W_p_j = np.array([10, 0, 0])

    # Velocities of agent i and j in the world coordinate frame for four time steps.
    W_v_i = [np.array([1, 1, 0]),
             np.array([0, 0, 0]),
             np.array([0, 1, 0])]

    W_v_j = [np.array([1, 0, 1]),
             np.array([1, 1, 0]),
             np.array([2, 0, 1])]

    # Initialize the realtive localization problem with a distance measurement
    # between the agents i and j.
    d_ij_init = np.linalg.norm(W_p_j - W_p_i)
    problem = Problem(d_ij_init)

    for k in range(len(W_v_i)):
        # Time step
        dt = 0.5

        # Euler forward integration of the positions of agent i and j.
        W_p_i = W_p_i + W_v_i[k] * dt
        W_p_j = W_p_j + W_v_j[k] * dt

        # Distance measurement between the agents i and j.
        d_ij = np.linalg.norm(W_p_j - W_p_i)

        # Local velocity measurement of the agents i and j.
        Li_v_i = T_WLi.T @ W_v_i[k]
        Lj_v_j = T_WLj.T @ W_v_j[k]

        problem.add_data(Li_v_i, Lj_v_j, d_ij, dt)

    # Solve the relative localization problem. Agent i localizes j in i' odometry frame.
    Li_d_est, heading_est, error = problem.solve()

    W_d_est = T_WLi @ Li_d_est
    W_d_sim = W_p_j - W_p_i

    print("=======================================================================================")
    print(f"Simulated relative position in world frame: {W_d_sim[0]}, {W_d_sim[1]}, {W_d_sim[2]}")
    print(f"Simulated relative heading angle: {(heading_j - heading_i)*180/np.pi}°")
    print("=======================================================================================")
    print(f"Relative localization in odometry frame: {Li_d_est[0]}, {Li_d_est[1]}, {Li_d_est[2]}")
    print(f"Relative localization transformed to world frame: {W_d_est[0]}, {W_d_est[1]}, {W_d_est[2]}")
    print(f"Estimated relative heading angle: {heading_est*180/np.pi}°")
    print(f"Optimization function objective value: {error}")
    print("=======================================================================================")
