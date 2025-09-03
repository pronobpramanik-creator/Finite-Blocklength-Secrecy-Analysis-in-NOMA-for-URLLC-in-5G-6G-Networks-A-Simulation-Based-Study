# Finite Blocklength Secrecy Analysis in NOMA for URLLC

This repository contains Python code for Monte Carlo simulations of secrecy performance in **Non-Orthogonal Multiple Access (NOMA)** under **finite blocklength (FBL)** constraints, with applications to **URLLC in 5G/6G networks**.

The script generates five key figures for analysis:

1. **ESR vs SNR** (n=200, α=0.5)  
2. **SOP vs SNR** (n=200, α=0.5)  
3. **ESR vs Blocklength** (SNR=10 dB, α=0.5)  
4. **SOP vs Blocklength** (SNR=10 dB, α=0.5)  
5. **ESR vs Power Allocation α** (SNR=10 dB, n=200)  

All results are stored automatically in the `figures/` directory.

---

## ⚙️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/NOMA-FBL-Secrecy.git
cd NOMA-FBL-Secrecy
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the simulation script:

```bash
python noma_fbl_secrecy.py
```

The script will:
- Run Monte Carlo simulations  
- Generate all 5 figures  
- Save them in the `figures/` folder  

---

## 📂 Repository Structure

```
NOMA-FBL-Secrecy/
│── noma_fbl_secrecy.py   # Main simulation script
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
│── figures/              # Auto-generated plots
```

---

## 👤 Author

**Pronob Pramanik**  
📧 Contact: (add your email if you want)  
📌 Research focus: 5G/6G, URLLC, NOMA, Secrecy, Finite Blocklength Analysis  

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use and adapt with citation.
