"""
Finite Blocklength Secrecy Analysis in NOMA for URLLC
=====================================================

This script performs Monte Carlo simulations for secrecy performance
of a downlink NOMA system under finite blocklength constraints.

It generates five figures:
    1. ESR vs SNR
    2. SOP vs SNR
    3. ESR vs Blocklength
    4. SOP vs Blocklength
    5. ESR vs Power Allocation (α)

Author: Pronob Pramanik
Contact: pronob.pramanik@gmail.com
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri

# -------------------------------
# Simulation Function
# -------------------------------
def simulate_noma_secrecy(snr_dB, blocklength, alpha, trials=50000, eps=1e-5):
    N0 = 1.0
    q_inv = ndtri(1 - eps)
    ESR_vals = []
    SOP_vals = []

    # Channel mean powers (Rayleigh fading -> exponential)
    scale_m = 1.0   # strong user
    scale_n = 0.25  # weak user
    scale_e = 0.25  # eavesdropper

    for snr in snr_dB:
        P = 10**(snr/10) * N0
        hm2 = np.random.exponential(scale_m, size=trials)
        hn2 = np.random.exponential(scale_n, size=trials)
        he2 = np.random.exponential(scale_e, size=trials)

        # Legitimate user SNRs
        gamma_n = (P*(1-alpha)*hn2) / (P*alpha*hn2 + N0)
        gamma_m = (P*alpha*hm2) / N0
        # Eavesdropper SNRs
        gamma_en = (P*(1-alpha)*he2) / (P*alpha*he2 + N0)
        gamma_em = (P*alpha*he2) / N0

        # Shannon capacities
        Cn = np.log2(1 + gamma_n)
        Cm = np.log2(1 + gamma_m)
        Cen = np.log2(1 + gamma_en)
        Cem = np.log2(1 + gamma_em)

        # Dispersion terms
        Vn = 1 - 1/((1+gamma_n)**2)
        Vm = 1 - 1/((1+gamma_m)**2)

        # Finite blocklength rates
        Rn = np.maximum(0, Cn - q_inv*np.sqrt(Vn/blocklength)/np.log(2))
        Rm = np.maximum(0, Cm - q_inv*np.sqrt(Vm/blocklength)/np.log(2))

        # Secrecy rates
        S_n = np.maximum(Rn - Cen, 0)
        S_m = np.maximum(Rm - Cem, 0)
        S_total = S_n + S_m

        # Metrics
        ESR_vals.append(np.mean(S_total))
        SOP_vals.append(np.mean(S_total < 1.0))  # threshold 1 bit/use

    return np.array(ESR_vals), np.array(SOP_vals)

# -------------------------------
# Matplotlib Global Style
# -------------------------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11
})

# -------------------------------
# Figure 1: ESR vs SNR
# -------------------------------
snr_range = np.arange(0, 22, 2)
ESR, SOP = simulate_noma_secrecy(snr_range, blocklength=200, alpha=0.5)

plt.figure()
plt.plot(snr_range, ESR, 'o-', label='ESR')
plt.xlabel('Transmit SNR (dB)')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.title('Fig. 1. ESR vs SNR (n=200, α=0.5)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig1_ESR_vs_SNR_alpha0.5_n200.png", dpi=300)

# -------------------------------
# Figure 2: SOP vs SNR
# -------------------------------
plt.figure()
plt.plot(snr_range, SOP, 'x-', color='red', label='SOP')
plt.xlabel('Transmit SNR (dB)')
plt.ylabel('Secrecy Outage Probability')
plt.title('Fig. 2. SOP vs SNR (n=200, α=0.5)')
plt.ylim([0, 1])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig2_SOP_vs_SNR_alpha0.5_n200.png", dpi=300)

# -------------------------------
# Figure 3: ESR vs Blocklength
# -------------------------------
blocklengths = [100, 200, 300, 400]
ESR_vs_n = []
for n in blocklengths:
    ESR, _ = simulate_noma_secrecy([10], blocklength=n, alpha=0.5)
    ESR_vs_n.append(ESR[0])

plt.figure()
plt.plot(blocklengths, ESR_vs_n, 's-', label='ESR')
plt.xlabel('Blocklength (n)')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.title('Fig. 3. ESR vs Blocklength (SNR=10 dB, α=0.5)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig3_ESR_vs_n_SNR10_alpha0.5.png", dpi=300)

# -------------------------------
# Figure 4: SOP vs Blocklength
# -------------------------------
SOP_vs_n = []
for n in blocklengths:
    _, SOP = simulate_noma_secrecy([10], blocklength=n, alpha=0.5)
    SOP_vs_n.append(SOP[0])

plt.figure()
plt.plot(blocklengths, SOP_vs_n, 'd-', color='purple', label='SOP')
plt.xlabel('Blocklength (n)')
plt.ylabel('Secrecy Outage Probability')
plt.title('Fig. 4. SOP vs Blocklength (SNR=10 dB, α=0.5)')
plt.ylim([0, 1])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig4_SOP_vs_n_SNR10_alpha0.5.png", dpi=300)

# -------------------------------
# Figure 5: ESR vs Alpha
# -------------------------------
alphas = [0.2, 0.5, 0.8]
ESR_vs_alpha = []
for a in alphas:
    ESR, _ = simulate_noma_secrecy([10], blocklength=200, alpha=a)
    ESR_vs_alpha.append(ESR[0])

plt.figure()
plt.plot(alphas, ESR_vs_alpha, 'o-', label='ESR')
plt.xlabel('Power Allocation α')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.title('Fig. 5. ESR vs α (SNR=10 dB, n=200)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig5_ESR_vs_alpha_SNR10_n200.png", dpi=300)

plt.show()
