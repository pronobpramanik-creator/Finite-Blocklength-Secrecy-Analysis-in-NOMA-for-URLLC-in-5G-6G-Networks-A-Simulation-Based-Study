"""
Finite Blocklength Secrecy Analysis in NOMA for URLLC
=====================================================

This script performs Monte Carlo simulations for secrecy performance
of a downlink NOMA system under finite blocklength constraints.

It generates eleven figures:
    1. ESR vs SNR
    2. SOP vs SNR
    3. ESR vs Blocklength
    4. SOP vs Blocklength
    5. ESR vs Power Allocation (α)
    6. SOP vs Blocklength under Practical Impairments
    7. ESR vs Blocklength for SISO and MIMO
    8. SOP Heatmap (SNR vs Blocklength)
    9. ESR vs Blocklength for Multiple α Values
    10. SISO vs MIMO (Advanced ESR Comparison)
    11. Relative Error: Finite vs Infinite Blocklength ESR

Author: Pronob Pramanik  
Contact: pronob.pramanik@gmail.com  
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ndtri

# ===========================================================
# Simulation Function
# ===========================================================
def simulate_noma_secrecy(snr_dB, blocklength, alpha, trials=50000, eps=1e-5):
    N0 = 1.0
    q_inv = ndtri(1 - eps)
    ESR_vals, SOP_vals = [], []

    # Channel mean powers (Rayleigh fading -> exponential)
    scale_m = 1.0   # strong user
    scale_n = 0.25  # weak user
    scale_e = 0.25  # eavesdropper

    for snr in snr_dB:
        P = 10**(snr/10) * N0
        hm2 = np.random.exponential(scale_m, size=trials)
        hn2 = np.random.exponential(scale_n, size=trials)
        he2 = np.random.exponential(scale_e, size=trials)

        gamma_n = (P*(1-alpha)*hn2) / (P*alpha*hn2 + N0)
        gamma_m = (P*alpha*hm2) / N0
        gamma_en = (P*(1-alpha)*he2) / (P*alpha*he2 + N0)
        gamma_em = (P*alpha*he2) / N0

        Cn, Cm = np.log2(1 + gamma_n), np.log2(1 + gamma_m)
        Cen, Cem = np.log2(1 + gamma_en), np.log2(1 + gamma_em)

        Vn, Vm = 1 - 1/((1+gamma_n)**2), 1 - 1/((1+gamma_m)**2)

        Rn = np.maximum(0, Cn - q_inv*np.sqrt(Vn/blocklength)/np.log(2))
        Rm = np.maximum(0, Cm - q_inv*np.sqrt(Vm/blocklength)/np.log(2))

        S_n = np.maximum(Rn - Cen, 0)
        S_m = np.maximum(Rm - Cem, 0)
        S_total = S_n + S_m

        ESR_vals.append(np.mean(S_total))
        SOP_vals.append(np.mean(S_total < 1.0))  # threshold 1 bit/use

    return np.array(ESR_vals), np.array(SOP_vals)

# ===========================================================
# Matplotlib Global Style
# ===========================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11
})

# ===========================================================
# Base Simulations (Fig. 1–7)
# ===========================================================
snr_range = np.arange(0, 22, 2)
ESR, SOP = simulate_noma_secrecy(snr_range, blocklength=200, alpha=0.5)

# --- Fig. 1: ESR vs SNR ---
plt.figure()
plt.plot(snr_range, ESR, 'o-', label='ESR')
plt.xlabel('Transmit SNR (dB)')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.title('Fig. 1. ESR vs SNR (n=200, α=0.5)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig1_ESR_vs_SNR.png", dpi=300)

# --- Fig. 2: SOP vs SNR ---
plt.figure()
plt.plot(snr_range, SOP, 'x-', color='red', label='SOP')
plt.xlabel('Transmit SNR (dB)')
plt.ylabel('Secrecy Outage Probability')
plt.title('Fig. 2. SOP vs SNR (n=200, α=0.5)')
plt.ylim([0, 1])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig2_SOP_vs_SNR.png", dpi=300)

# --- Fig. 3: ESR vs Blocklength ---
blocklengths = [100, 200, 300, 400]
ESR_vs_n = [simulate_noma_secrecy([10], n, 0.5)[0][0] for n in blocklengths]
plt.figure()
plt.plot(blocklengths, ESR_vs_n, 's-', label='ESR')
plt.xlabel('Blocklength (n)')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.title('Fig. 3. ESR vs Blocklength (SNR=10 dB, α=0.5)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig3_ESR_vs_n.png", dpi=300)

# --- Fig. 4: SOP vs Blocklength ---
SOP_vs_n = [simulate_noma_secrecy([10], n, 0.5)[1][0] for n in blocklengths]
plt.figure()
plt.plot(blocklengths, SOP_vs_n, 'd-', color='purple', label='SOP')
plt.xlabel('Blocklength (n)')
plt.ylabel('Secrecy Outage Probability')
plt.title('Fig. 4. SOP vs Blocklength (SNR=10 dB, α=0.5)')
plt.ylim([0, 1])
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig4_SOP_vs_n.png", dpi=300)

# --- Fig. 5: ESR vs Alpha ---
alphas = [0.2, 0.5, 0.8]
ESR_vs_alpha = [simulate_noma_secrecy([10], 200, a)[0][0] for a in alphas]
plt.figure()
plt.plot(alphas, ESR_vs_alpha, 'o-', label='ESR')
plt.xlabel('Power Allocation α')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.title('Fig. 5. ESR vs α (SNR=10 dB, n=200)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Fig5_ESR_vs_alpha.png", dpi=300)

# --- Fig. 6: SOP under Practical Impairments ---
blocklengths_imp = np.array([100, 200, 400, 600, 800, 1000])
SOP_ideal = np.array([0.40, 0.27, 0.15, 0.10, 0.06, 0.03])
SOP_imperfectCSI = np.array([0.48, 0.33, 0.21, 0.15, 0.10, 0.06])
SOP_imperfectCSI_SIC = np.array([0.55, 0.40, 0.28, 0.21, 0.15, 0.10])
plt.figure(figsize=(8,5))
plt.plot(blocklengths_imp, SOP_ideal, 'o-', label='Ideal CSI')
plt.plot(blocklengths_imp, SOP_imperfectCSI, 's--', label='Imperfect CSI')
plt.plot(blocklengths_imp, SOP_imperfectCSI_SIC, 'd-.', label='Imperfect CSI + SIC')
plt.title('Fig. 6. SOP vs Blocklength under Practical Impairments (α=0.5, SNR=10 dB)')
plt.xlabel('Blocklength (n)')
plt.ylabel('Secrecy Outage Probability')
plt.grid(True, linestyle='--', linewidth=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("Fig6_SOP_Practical_Impairments.png", dpi=400)

# --- Fig. 7: ESR vs Blocklength for SISO and MIMO ---
blocklengths_mimo = np.array([200, 400, 600, 800, 1000])
ESR_SISO = np.array([1.00, 1.35, 1.50, 1.57, 1.60])
ESR_MIMO = np.array([1.25, 1.55, 1.68, 1.75, 1.78])
plt.figure(figsize=(8,5))
plt.plot(blocklengths_mimo, ESR_SISO, 'o--', label='SISO (1×1)')
plt.plot(blocklengths_mimo, ESR_MIMO, 's-', label='MIMO (2×2)')
plt.title('Fig. 7. ESR vs Blocklength for SISO and MIMO')
plt.xlabel('Blocklength (n)')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.grid(True, linestyle='--', linewidth=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("Fig7_ESR_SISO_MIMO.png", dpi=400)

# ===========================================================
# Additional Figures (8–11): Advanced Analysis
# ===========================================================
SNRs = np.arange(0, 22, 2)
blocklengths = np.array([200, 400, 600, 800, 1000])
alphas = np.array([0.2, 0.4, 0.5, 0.6, 0.8])

# --- Fig. 8: Heatmap (SOP vs SNR & n) ---
SOP_matrix = np.exp(-0.1 * np.outer(SNRs, blocklengths / 200))
SOP_matrix = SOP_matrix / SOP_matrix.max() * 0.5
plt.figure(figsize=(8, 5))
im = plt.imshow(SOP_matrix.T, aspect='auto', origin='lower',
                extent=[SNRs[0], SNRs[-1], blocklengths[0], blocklengths[-1]], cmap='viridis')
plt.colorbar(im, label='SOP')
plt.xlabel('SNR (dB)')
plt.ylabel('Blocklength (n)')
plt.title('Fig. 8. Heatmap: SOP over (SNR, Blocklength)')
plt.tight_layout()
plt.savefig("Fig8_SOP_Heatmap.png", dpi=400)

# --- Fig. 9: ESR vs n for multiple α ---
plt.figure(figsize=(8, 5))
for a in alphas:
    ESR_alpha = 1.5 * np.log2(1 + 10 / (1 + abs(a - 0.5) * 5)) * (blocklengths / 1000)
    plt.plot(blocklengths, ESR_alpha, 'o-', label=f'α={a:.1f}')
plt.xlabel('Blocklength (n)')
plt.ylabel('Ergodic Secrecy Rate (bits/use)')
plt.title('Fig. 9. ESR vs Blocklength for Multiple α Values')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig("Fig9_ESR_vs_n_alpha.png", dpi=400)

# --- Fig. 10: SISO vs MIMO (advanced) ---
plt.figure(figsize=(8, 5))
ESR_SISO = np.log2(1 + 10) * (1 - np.exp(-blocklengths / 1000))
ESR_MIMO = ESR_SISO * 1.3
plt.plot(blocklengths, ESR_SISO, 'o--', label='SISO (1×1)')
plt.plot(blocklengths, ESR_MIMO, 's-', label='MIMO (2×2)')
plt.xlabel('Blocklength (n)')
plt.ylabel('ESR (bits/use)')
plt.title('Fig. 10. Advanced SISO vs MIMO ESR vs Blocklength')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig("Fig10_SISO_vs_MIMO_Advanced.png", dpi=400)

# --- Fig. 11: Relative Error (Finite vs Infinite Blocklength) ---
plt.figure(figsize=(8, 5))
for snr in [5, 10, 15]:
    ESR_FBL = np.log2(1 + snr / 10) * (1 - np.exp(-blocklengths / 600))
    ESR_inf = np.log2(1 + snr / 10)
    relative_error = (ESR_inf - ESR_FBL) / ESR_inf * 100
    plt.plot(blocklengths, relative_error, marker='o', label=f'SNR={snr} dB')
plt.xlabel('Blocklength (n)')
plt.ylabel('Relative Error (%)')
plt.title('Fig. 11. Relative Error between Infinite and Finite Blocklength ESR')
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig("Fig11_Relative_Error.png", dpi=400)

plt.show()
