# -*- coding: utf-8 -*-
"""Neural-network steering controller trained on the clock steering model.

The original notebook used a fixed LQG controller.  Here we reuse the same
stochastic model but replace the controller with a neural network trained in a
closed loop, similar in spirit to the NN PID in nnpid_red2.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def covar_by_noise(q, dt):
    """Q matrix for the integrated noise components."""
    q1, q2 = q[1], q[2]
    return np.array(
        [
            [q1**2 * dt + q2**2 * dt**3 / 3.0, q2**2 * dt**2 / 2.0],
            [q2**2 * dt**2 / 2.0, q2**2 * dt],
        ],
        dtype=np.double,
    )


def allan_deviation(z, dt, tau):
    """Classical Allan deviation."""
    adev = np.zeros(tau.size, dtype="double")
    n = z.size
    maxi = 0
    for i in range(tau.size):
        if tau[i] * 3 < n:
            maxi = i
            sigma2 = np.sum(
                (z[2 * tau[i] :: 1] - 2 * z[tau[i] : -tau[i] : 1] + z[0:-2 * tau[i] : 1]) ** 2
            )
            adev[i] = np.sqrt(0.5 * sigma2 / (n - 2 * tau[i])) / tau[i] / dt
        else:
            break
    return tau[:maxi].astype(np.double) * dt, adev[:maxi]


def parabolic_deviation(z, dt, tau):
    """Parabolic (Hadamard) deviation."""
    adev = np.zeros(tau.size, dtype="double")
    n = z.size
    maxi = 0
    for i in range(tau.size):
        if tau[i] * 3 < n:
            maxi = i
            m = 0
            s = 0
            c1 = np.polyfit(range(0, tau[i] + 1, 1), z[0 : tau[i] + 1 : 1], 1)
            for j in range(tau[i], n - tau[i], tau[i]):
                c2 = np.polyfit(range(j, j + tau[i] + 1, 1), z[j : j + tau[i] + 1 : 1], 1)
                s += (c1[0] - c2[0]) ** 2
                m += 1
                c1 = c2
            adev[i] = np.sqrt(0.5 * s / m) / dt
        else:
            break
    return tau[:maxi].astype(np.double) * dt, adev[:maxi]


class NeuralSteeringController(nn.Module):
    """Small fully-connected controller that receives Kalman estimates."""

    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, phase_est, freq_est, meas):
        features = torch.stack((phase_est, freq_est, meas), dim=-1)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        output = self.net(features).squeeze(-1)
        if output.shape[0] == 1:
            return output.squeeze(0)
        return output


class SteeringSimulator:
    """Clock steering model with GNSS reference noise."""

    def __init__(self, free_noise, ref_noise, dt, ctrl_interval, drift, device="cpu"):
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.dt = dt
        self.ctrl_interval = ctrl_interval
        self.drift = drift

        self.F = torch.tensor([[1.0, dt], [0.0, 1.0]], dtype=self.dtype, device=self.device)
        self.B = torch.tensor([dt, 1.0], dtype=self.dtype, device=self.device)
        self.H = torch.tensor([[1.0, 0.0]], dtype=self.dtype, device=self.device)
        self.I = torch.eye(2, dtype=self.dtype, device=self.device)
        self.D = torch.tensor([0.5 * dt**2, dt], dtype=self.dtype, device=self.device)
        self.drift_vec = self.D * drift

        free_q = covar_by_noise(free_noise, dt)
        ref_q = covar_by_noise(ref_noise, dt)
        self.free_Q = torch.tensor(free_q, dtype=self.dtype, device=self.device)
        self.ref_Q = torch.tensor(ref_q, dtype=self.dtype, device=self.device)
        self.free_L = torch.linalg.cholesky(self.free_Q)
        self.ref_L = torch.linalg.cholesky(self.ref_Q)

        self.phase_noise_std = torch.tensor(free_noise[0], dtype=self.dtype, device=self.device)
        self.ref_phase_noise_std = torch.tensor(ref_noise[0], dtype=self.dtype, device=self.device)
        self.freq_noise_std = torch.tensor(free_noise[1], dtype=self.dtype, device=self.device)
        self.R = torch.tensor(
            free_noise[0] ** 2 + ref_noise[0] ** 2, dtype=self.dtype, device=self.device
        )

    def run(self, controller, steps, training=True, phase_jump_step=None, phase_jump_value=0.0):
        if training:
            controller.train()
        else:
            controller.eval()

        phase_jump_value = torch.tensor(phase_jump_value, dtype=self.dtype, device=self.device)

        Xlock = torch.zeros(2, dtype=self.dtype, device=self.device)
        Xfree = torch.zeros(2, dtype=self.dtype, device=self.device)
        Xref = torch.zeros(2, dtype=self.dtype, device=self.device)
        dX = torch.zeros(2, dtype=self.dtype, device=self.device)
        P = torch.tensor(
            [
                [self.R.item(), 0.0],
                [0.0, (self.freq_noise_std**2).item()],
            ],
            dtype=self.dtype,
            device=self.device,
        )
        u_prev = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        history = {
            "phase_diff_true": [],
            "freq_diff_true": [],
            "phase_est": [],
            "freq_est": [],
            "measurements": [],
            "locked_output": [],
            "free_output": [],
            "ref_output": [],
            "control_applied": [],
            "control_commands": [],
        }

        for step in range(steps):
            free_w = torch.randn(2, dtype=self.dtype, device=self.device)
            ref_w = torch.randn(2, dtype=self.dtype, device=self.device)

            Xlock = self.F @ Xlock + self.free_L @ free_w + self.B * u_prev + self.drift_vec
            Xfree = self.F @ Xfree + self.free_L @ free_w + self.drift_vec
            Xref = self.F @ Xref + self.ref_L @ ref_w

            if phase_jump_step is not None and step == phase_jump_step:
                Xlock = Xlock.clone()
                Xlock[0] = Xlock[0] + phase_jump_value

            wpn_free = torch.randn((), dtype=self.dtype, device=self.device) * self.phase_noise_std
            wpn_ref = torch.randn((), dtype=self.dtype, device=self.device) * self.ref_phase_noise_std

            locked_output = Xlock[0] + wpn_free
            free_output = Xfree[0] + wpn_free
            ref_output = Xref[0] + wpn_ref
            z = locked_output - ref_output

            Ppred = self.F @ P @ self.F.T + self.free_Q + self.ref_Q
            dXpred = self.F @ dX + self.B * u_prev
            S = (self.H @ Ppred @ self.H.T).squeeze() + self.R
            K = (Ppred @ self.H.T) / S
            innov = z - (self.H @ dXpred.unsqueeze(-1)).squeeze()
            dX = dXpred + (K.squeeze() * innov)
            P = (self.I - K @ self.H) @ Ppred

            phase_diff_true = Xlock[0] - Xref[0]
            freq_diff_true = Xlock[1] - Xref[1]

            history["phase_diff_true"].append(phase_diff_true)
            history["freq_diff_true"].append(freq_diff_true)
            history["phase_est"].append(dX[0])
            history["freq_est"].append(dX[1])
            history["measurements"].append(z)
            history["locked_output"].append(locked_output)
            history["free_output"].append(free_output)
            history["ref_output"].append(ref_output)
            history["control_applied"].append(u_prev)

            if step % self.ctrl_interval == 0:
                cmd = controller(dX[0]*1e8, dX[1]*1e13, torch.abs(innov)*1e3)*1e-12
            else:
                cmd = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            history["control_commands"].append(cmd)
            u_prev = cmd #-1.90165607e-07*dX[0] - 1.90169990e-02*dX[1]#cmd

        return {k: torch.stack(v) for k, v in history.items()}


def train_controller(
    controller,
    simulator,
    num_epochs=50,
    steps_per_epoch=200,
    phase_weight=1e-3,
    freq_weight=1,
    control_weight=1e10,
):
    optimizer = optim.Adam(controller.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        result = simulator.run(controller, steps_per_epoch, training=True)
        phase_loss = (result["phase_diff_true"] ** 2).mean()
        freq_loss = (result["freq_diff_true"] ** 2).mean()
        control_loss = (result["control_commands"] ** 2).mean()
        loss = phase_weight * phase_loss + freq_weight * freq_loss + control_weight * control_loss
        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            print(
                f"Epoch {epoch + 1:03d}: "
                f"phase_loss={phase_loss.item():.3e}, "
                f"freq_loss={freq_loss.item():.3e}, "
                f"control_loss={control_loss.item():.3e}"
            )


def main():
    torch.manual_seed(0)

    GNSS_noise = (np.sqrt(1e-18), 1e-50, 1e-60)
    q1 = 1e-25
    q2 = 2.3e-34
    free_noise = (np.sqrt(9e-26), np.sqrt(q1), np.sqrt(q2))

    dt = 3600
    ctrl_interval = 1
    drift = 5e-16 / 86400

    simulator = SteeringSimulator(
        free_noise=free_noise,
        ref_noise=GNSS_noise,
        dt=dt,
        ctrl_interval=ctrl_interval,
        drift=drift,
    )

    controller = NeuralSteeringController(hidden_size=64).to(simulator.device).double()

    print("Training neural controller...")
    train_controller(controller, simulator)

    print("Running evaluation...")
    steps = 10000
    jump_step = None #steps // 2
    phase_jump = 0 #5e-10
    with torch.no_grad():
        result = simulator.run(
            controller,
            steps=steps,
            training=False,
            phase_jump_step=jump_step,
            phase_jump_value=phase_jump,
        )

    z = result["measurements"].cpu().numpy()
    x = result["locked_output"].cpu().numpy()
    xK = result["phase_est"].cpu().numpy()
    yK = result["freq_est"].cpu().numpy()
    u = result["control_commands"].cpu().numpy()
    free = result["free_output"].cpu().numpy()
    ref = result["ref_output"].cpu().numpy()

    plt.figure(1)
    plt.plot(z, label="Измеренная разность фаз")
    plt.plot(x, "y", label="Фаза локального осц.")
    plt.plot(xK, "r", label="Оценка Калмана")
    plt.ylabel("Разность фаз, с")
    plt.xlabel("Время, такты")
    plt.legend()

    plt.figure(2)
    t0 = 100
    plt.plot(yK[t0:], "b", label="Разность частот (Калман)")
    plt.plot(u[t0:], "r", label="Управляющее воздействие")
    plt.legend()
    plt.xlabel("Время, такты")

    tau = np.arange(1, 10)
    tau = np.append(tau, np.arange(10, 100, 10))
    tau = np.append(tau, np.arange(100, 1000, 100))
    tau = np.append(tau, np.arange(1000, 10000, 1000))
    tau = np.append(tau, np.arange(10000, 100000, 10000))
    tau = np.append(tau, np.arange(100000, 1000000, 100000))
    tau = np.append(tau, np.arange(1000000, 10000000, 1000000))

    taus, adev_lock = allan_deviation(x[1000:], dt, tau)
    _, adev_free = allan_deviation(free[1000:], dt, tau)
    _, adev_ref = allan_deviation(ref[1000:], dt, tau)

    plt.figure(3)
    plt.loglog(taus, adev_free, label="Свободный")
    plt.loglog(taus, adev_ref, label="Опорный")
    plt.loglog(taus, adev_lock, label="Подстройка NN")
    plt.xlabel("Интервал времени измерения, с")
    plt.ylabel("ADEV")
    plt.legend()

    taus_p, pdev_lock = parabolic_deviation(x[1000:], dt, tau)
    _, pdev_free = parabolic_deviation(free[1000:], dt, tau)
    _, pdev_ref = parabolic_deviation(ref[1000:], dt, tau)

    plt.figure(4)
    plt.loglog(taus_p, pdev_free, label="Свободный")
    plt.loglog(taus_p, pdev_ref, label="Опорный")
    plt.loglog(taus_p, pdev_lock, label="Подстройка NN")
    plt.xlabel("Интервал времени измерения, с")
    plt.ylabel("PDEV")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
