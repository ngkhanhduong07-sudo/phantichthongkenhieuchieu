import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CHUONG 3: TIEN XU LY DU LIEU VA LAP TRINH TINH TOAN TREN PYTHON
# 3.1 Lam sach & xu ly ngoai lai
# 3.2 Tinh dac trung mau (X_bar, S, R, |S|)
# 3.3 Kiem tra cac gia dinh thong ke (Mardia, Box's M)
# 3.4 Thuc thi kiem dinh Hotelling T^2
# ============================================================


# ============================================================
# 1. DOC DU LIEU
# ============================================================

du_lieu = pd.read_csv("credit_risk_dataset.csv")

# 5 bien tai chinh dung trong phan tich (dung ten cot that trong file du lieu)
bien_tai_chinh = ["person_income", "loan_amnt", "loan_int_rate",
                   "loan_percent_income", "cb_person_cred_hist_length"]

# Bien phan nhom: 0 = tra dung han, 1 = vo no
bien_nhom = "loan_status"

# Muc y nghia dung xuyen suot bao cao
muc_y_nghia = 0.05

# B1. Loai gia tri bat hop ly do loi nhap lieu (THEM MOI)
n_goc = len(du_lieu)
du_lieu = du_lieu[(du_lieu["person_age"] <= 100) & (du_lieu["person_emp_length"] <= 60)].copy()
print(f"B1. Loai gia tri bat hop ly (tuoi>100, tham nien>60 nam): -{n_goc - len(du_lieu)} dong")

# Chi giu cac cot can thiet va loai cac dong bi thieu du lieu
du_lieu_phan_tich = du_lieu[bien_tai_chinh + [bien_nhom]].copy()
du_lieu_phan_tich = du_lieu_phan_tich.dropna()

print("So quan sat sau khi loai du lieu thieu:", len(du_lieu_phan_tich))
print(du_lieu_phan_tich[bien_nhom].value_counts())


# ============================================================
# 3.1. LAM SACH DU LIEU VA XU LY NGOAI LAI
#       BANG KHOANG CACH MAHALANOBIS  (giong TH4/TH5)
# ============================================================

print("\n============================================")
print("3.1. XU LY NGOAI LAI BANG MAHALANOBIS")
print("============================================")

X_tho = du_lieu_phan_tich[bien_tai_chinh].to_numpy()
p = len(bien_tai_chinh)

# Vector trung binh va ma tran hiep phuong sai cua TOAN BO mau tho
vector_trung_binh_tho = X_tho.mean(axis=0)
S_tho = np.cov(X_tho, rowvar=False, ddof=1)

# Chan doan tinh hop le cua S truoc khi nghich dao (giong Lab1 cua TH6)
eigvals_S_tho = np.linalg.eigvalsh(S_tho)
print(f"S doi xung: {np.allclose(S_tho, S_tho.T)} | Xac dinh duong: {np.all(eigvals_S_tho > 0)}")
print(f"So dieu kien cond(S) = {np.linalg.cond(S_tho):.4f}")

S_tho_nghich_dao = np.linalg.inv(S_tho)  # giu lai bien nay de tham khao/doi chieu

# Khoang cach Mahalanobis binh phuong: D^2_i = (x_i - xbar)' S^-1 (x_i - xbar)
# SUA: dung np.linalg.solve thay vi nhan qua ma tran nghich dao truc tiep,
# giong khuyen nghi cua TH5/TH6 (on dinh so hoc hon), thay cho np.einsum.
hieu = X_tho - vector_trung_binh_tho
Z_tho = np.linalg.solve(S_tho, hieu.T).T          # Z = S^{-1} * hieu, vector hoa
mahalanobis_binh_phuong = np.sum(Z_tho * hieu, axis=1)

# Nguong toi han theo phan phoi Chi-binh-phuong bac tu do p, tai muc y nghia da chon
nguong_mahalanobis = chi2.ppf(1 - muc_y_nghia, df=p)

la_ngoai_lai = mahalanobis_binh_phuong > nguong_mahalanobis
so_ngoai_lai = int(la_ngoai_lai.sum())

print(f"Muc y nghia dung de xac dinh nguong: alpha = {muc_y_nghia}")
print(f"Nguong Mahalanobis D^2 = chi2.ppf(1-alpha, df={p}) = {nguong_mahalanobis:.4f}")
print("So quan sat bi loai (ngoai lai):", so_ngoai_lai)

du_lieu_sach = du_lieu_phan_tich[~la_ngoai_lai].copy()
print("So quan sat con lai sau khi loai ngoai lai:", len(du_lieu_sach))
print(du_lieu_sach[bien_nhom].value_counts())

ten_file_dulieu_sach = "credit_risk_dataset_sach.csv"
du_lieu_sach.to_csv(ten_file_dulieu_sach, index=False, encoding="utf-8-sig")
print("Da luu du lieu sau khi lam sach vao:", ten_file_dulieu_sach)


# ============================================================
# 3.2. TINH DAC TRUNG MAU: X_bar, MA TRAN S, MA TRAN R, |S|
#       (theo dung yeu cau muc 3.2 cua de cuong tieu luan)
# ============================================================

print("\n============================================")
print("3.2. TINH DAC TRUNG MAU (X_bar, S, R, |S|)")
print("============================================")

X_sach_toanbo = du_lieu_sach[bien_tai_chinh].to_numpy()

# Vector trung binh mau chung (sau khi da lam sach)
xbar_chung = X_sach_toanbo.mean(axis=0)
print("\nVector trung binh mau chung X_bar:")
for ten, gt in zip(bien_tai_chinh, xbar_chung):
    print(f"  {ten}: {gt:.4f}")

# Ma tran hiep phuong sai mau S
S_chung = np.cov(X_sach_toanbo, rowvar=False, ddof=1)
print("\nMa tran hiep phuong sai mau S:")
print(np.round(S_chung, 4))

# Dinh thuc |S| (phuong sai tong quat - generalized variance)
det_S = np.linalg.det(S_chung)
print(f"\nDinh thuc |S| (phuong sai tong quat) = {det_S:.6e}")

# Ma tran he so tuong quan mau R
do_lech_chuan = np.sqrt(np.diag(S_chung))
R_chung = S_chung / np.outer(do_lech_chuan, do_lech_chuan)
print("\nMa tran he so tuong quan mau R:")
print(np.round(R_chung, 4))

# Vector trung binh rieng cho tung nhom (dung lai o muc 3.4)
nhom_0_so = du_lieu_sach[du_lieu_sach[bien_nhom] == 0][bien_tai_chinh].to_numpy()
nhom_1_so = du_lieu_sach[du_lieu_sach[bien_nhom] == 1][bien_tai_chinh].to_numpy()
print("\nVector trung binh nhom 0 (tra dung han):")
print(np.round(nhom_0_so.mean(axis=0), 4))
print("Vector trung binh nhom 1 (vo no):")
print(np.round(nhom_1_so.mean(axis=0), 4))


# ============================================================
# 3.3. KIEM TRA CAC GIA DINH THONG KE
#       3.3.1 Tinh chuan nhieu chieu - Mardia
#       3.3.2 Dong nhat ma tran hiep phuong sai - Box's M
# ============================================================

print("\n============================================")
print("3.3. KIEM TRA CAC GIA DINH THONG KE")
print("============================================")

# ------------------------------------------------------------
# 3.3.1. KIEM DINH CHUAN NHIEU CHIEU - MARDIA
# ------------------------------------------------------------

print("\n--- 3.3.1. KIEM DINH MARDIA ---")

so_mau_mardia = min(5000, len(X_sach_toanbo))
rng = np.random.default_rng(42)

if len(X_sach_toanbo) > so_mau_mardia:
    chi_so = rng.choice(len(X_sach_toanbo), size=so_mau_mardia, replace=False)
    X_mardia = X_sach_toanbo[chi_so]
else:
    X_mardia = X_sach_toanbo

n_mardia = len(X_mardia)
print("So quan sat dung cho Mardia =", n_mardia)

X_trung_tam = X_mardia - X_mardia.mean(axis=0)
S_mardia = np.cov(X_mardia, rowvar=False, ddof=1)
# SUA: dung np.linalg.solve thay vi np.linalg.inv (dong bo phong cach TH5/TH6/TH7)
Z_mardia = np.linalg.solve(S_mardia, X_trung_tam.T)   # Z = S^{-1} * X_trung_tam^T
Q = X_trung_tam @ Z_mardia
D2_mardia = np.diag(Q)

# Mardia Skewness
b1p = np.sum(Q ** 3) / (n_mardia ** 2)
skew_stat = n_mardia * b1p / 6
df_skew = p * (p + 1) * (p + 2) / 6
p_value_skew = stats.chi2.sf(skew_stat, df_skew)

# Mardia Kurtosis
b2p = np.sum(D2_mardia ** 2) / n_mardia
expected_kurtosis = p * (p + 2)
variance_kurtosis = 8 * p * (p + 2) / n_mardia
z_kurt = (b2p - expected_kurtosis) / np.sqrt(variance_kurtosis)
p_value_kurt = 2 * stats.norm.sf(abs(z_kurt))

print(f"Mardia Skewness statistic = {skew_stat:.4f}, df = {df_skew:.1f}, p-value = {p_value_skew:.6f}")
print(f"Mardia Kurtosis statistic (b2p) = {b2p:.4f}, Z = {z_kurt:.4f}, p-value = {p_value_kurt:.6f}")

if p_value_skew < muc_y_nghia or p_value_kurt < muc_y_nghia:
    print("Ket luan: Bac bo H0 -> du lieu khong phu hop voi phan phoi chuan nhieu chieu.")
else:
    print("Ket luan: Chua du bang chung bac bo H0 -> chua co bang chung du lieu khong chuan nhieu chieu.")

# ------------------------------------------------------------
# 3.3.2. KIEM DINH DONG NHAT MA TRAN HIEP PHUONG SAI - BOX'S M
# ------------------------------------------------------------

print("\n--- 3.3.2. KIEM DINH BOX'S M ---")

nhom_0 = nhom_0_so
nhom_1 = nhom_1_so
n_0 = len(nhom_0)
n_1 = len(nhom_1)

S_0 = np.cov(nhom_0, rowvar=False, ddof=1)
S_1 = np.cov(nhom_1, rowvar=False, ddof=1)

S_p = ((n_0 - 1) * S_0 + (n_1 - 1) * S_1) / (n_0 + n_1 - 2)

M = ((n_0 + n_1 - 2) * np.log(np.linalg.det(S_p))
     - (n_0 - 1) * np.log(np.linalg.det(S_0))
     - (n_1 - 1) * np.log(np.linalg.det(S_1)))

g = 2
sum_inv = (1 / (n_0 - 1) + 1 / (n_1 - 1) - 1 / (n_0 + n_1 - 2))
C = ((2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) * sum_inv

# SUA: cong thuc F cu (F_box/df1_box/df2_box) khong dung chuan Box's M (1949).
# Thay bang xap xi Chi-square don gian, dung va de kiem chung hon:
#   Chi_stat = M * (1 - C)  ~  Chi-square(df = (g-1)*p*(p+1)/2)
chi_stat_box = M * (1 - C)
df_box = (g - 1) * p * (p + 1) / 2
p_value_box = stats.chi2.sf(chi_stat_box, df_box)

print(f"n0 (tra dung han) = {n_0}, n1 (vo no) = {n_1}")
print(f"Box's M = {M:.4f}")
print(f"Chi-square xap xi = {chi_stat_box:.4f}, df = {df_box:.0f}")
print(f"p-value = {p_value_box:.6f}")

if p_value_box < muc_y_nghia:
    print("Ket luan: Bac bo H0 -> ma tran hiep phuong sai khong dong nhat.")
else:
    print("Ket luan: Chua du bang chung bac bo H0 -> co the chap nhan gia dinh dong nhat ma tran hiep phuong sai.")


# ============================================================
# 3.4. LAP TRINH TINH TOAN KIEM DINH HOTELLING T^2 (2 MAU DOC LAP)
#       Dong goi thanh ham dung theo dung khuon mau TH7
#       (ham hotelling_one_sample) va dung np.linalg.solve
#       thay vi np.linalg.inv, giong cach lam cua TH6/TH7.
# ============================================================

print("\n============================================")
print("3.4. KIEM DINH HOTELLING T^2 - 2 MAU DOC LAP")
print("============================================")


def hotelling_two_sample(X0, X1, alpha=0.05):
    """
    Kiem dinh Hotelling T^2 cho hai mau doc lap:
        H0: mu0 = mu1   vs   H1: mu0 != mu1
    Dung khi ma tran hiep phuong sai tong the chua biet nhung duoc
    gia dinh la dong nhat giua hai nhom (dung Sp gop).
    Cach viet phong theo ham hotelling_one_sample cua TH7.
    """
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float)
    n0, p0 = X0.shape
    n1, p1 = X1.shape
    if p0 != p1:
        raise ValueError("Hai nhom phai co cung so bien p")
    p = p0
    N = n0 + n1

    if N - p - 1 <= 0:
        raise ValueError(f"Dieu kien N - p - 1 > 0 khong thoa man ({N - p - 1} <= 0)")

    xbar0 = X0.mean(axis=0)
    xbar1 = X1.mean(axis=0)
    S0 = np.cov(X0, rowvar=False, ddof=1)
    S1 = np.cov(X1, rowvar=False, ddof=1)

    # Ma tran hiep phuong sai gop Sp = [(n0-1)S0 + (n1-1)S1] / (N-2)
    Sp = ((n0 - 1) * S0 + (n1 - 1) * S1) / (N - 2)

    # Vector sai khac d = xbar0 - xbar1
    d = xbar0 - xbar1

    # Giai he phuong trinh Sp * z = d  =>  z = Sp^(-1) * d  (giong TH6/TH7)
    z = np.linalg.solve(Sp, d)

    # Thong ke Hotelling T^2 = (n0*n1/N) * d^T * z
    T2 = float((n0 * n1 / N) * (d @ z))

    # Quy doi T^2 sang F: F = (N-p-1)/(p*(N-2)) * T^2 ~ F(p, N-p-1)
    F_val = float((N - p - 1) / (p * (N - 2)) * T2)
    crit = float(stats.f.ppf(1 - alpha, dfn=p, dfd=N - p - 1))
    pval = float(stats.f.sf(F_val, dfn=p, dfd=N - p - 1))

    return {
        "n0": n0, "n1": n1, "p": p, "N": N,
        "xbar0": xbar0, "xbar1": xbar1, "Sp": Sp,
        "T2": T2, "F_stat": F_val, "crit_F": crit,
        "p_value": pval, "reject_H0": F_val > crit,
    }


ket_qua_hotelling = hotelling_two_sample(nhom_0, nhom_1, alpha=muc_y_nghia)

print("\nVector trung binh nhom 0 (tra dung han):")
print(np.round(ket_qua_hotelling["xbar0"], 4))
print("\nVector trung binh nhom 1 (vo no):")
print(np.round(ket_qua_hotelling["xbar1"], 4))

print("\nMa tran hiep phuong sai gop S_p:")
print(np.round(ket_qua_hotelling["Sp"], 4))

print(f"\nHotelling T^2 = {ket_qua_hotelling['T2']:.4f}")
print(f"F = {ket_qua_hotelling['F_stat']:.4f}")
print(f"Gia tri toi han F_{ket_qua_hotelling['p']},"
      f"{ket_qua_hotelling['N']-ket_qua_hotelling['p']-1}"
      f"({1-muc_y_nghia:.2f}) = {ket_qua_hotelling['crit_F']:.4f}")
print(f"Bac tu do F = ({ket_qua_hotelling['p']}, "
      f"{ket_qua_hotelling['N']-ket_qua_hotelling['p']-1})")
print(f"p-value = {ket_qua_hotelling['p_value']:.6g}")

print("\n============================================")
print("KET LUAN KIEM DINH HOTELLING T^2")
print("============================================")
if ket_qua_hotelling["reject_H0"]:
    print(f"Bac bo H0 o muc y nghia {muc_y_nghia}.")
    print("Vector trung binh cua 5 bien tai chinh giua hai nhom khac nhau co y nghia thong ke.")
else:
    print(f"Chua du bang chung bac bo H0 o muc y nghia {muc_y_nghia}.")

# Cac bien dung lai o Chuong 4 (giu ten cu de khong pha vo phan ve bieu do)
S_p = ket_qua_hotelling["Sp"]


# ============================================================
# CHUONG 4: TRUC QUAN HOA DU LIEU DA BIEN (VE BIEU DO)
# ============================================================
print("\n============================================")
print("DANG TIEN HANH VE VA LUU BIEU DO CHO CHUONG 4...")
print("============================================")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# 1. Heatmap ma tran tuong quan (dung lai R da tinh o muc 3.2)
plt.figure(figsize=(8, 6))
sns.heatmap(
    pd.DataFrame(R_chung, index=bien_tai_chinh, columns=bien_tai_chinh),
    annot=True, fmt=".2f", cmap="coolwarm", square=True,
    linewidths=0.5, vmin=-1, vmax=1,
)
plt.title("Ma tran he so tuong quan Pearson (R)", fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig("heatmap_tuong_quan.png", dpi=300)
plt.close()
print("-> Da luu: heatmap_tuong_quan.png")


# 2. Pairplot (KHONG nhung elip - xem giai thich va hinh rieng elip_tin_cay_95.png ben duoi)
# SUA: ham cu "draw_confidence_ellipse" ve elip CHUA 95% DU LIEU (dung Chi-square,
# df=2, n_std=2.447) -> khac hoan toan y nghia "elip TIN CAY 95% cho VECTOR TRUNG BINH"
# ma de bai yeu cau. Voi N lon (hang chuc nghin), 2 loai elip nay khac nhau rat nhieu:
# elip du lieu luon to va chong lan (vi bien dong ca nhan lon), trong khi elip tin cay
# cho trung binh se rat nho va tach biet ro rang (dung theo phan phoi F).
def draw_mean_confidence_ellipse(x, y, ax, **kwargs):
    """Ve elip TIN CAY 95% cho VECTOR TRUNG BINH 2 chieu (khac elip du lieu).
    Cong thuc: c^2 = 2(n-1) / (n(n-2)) * F_(0.95; 2, n-2)  (giong TH6/TH7: dung
    phan phoi F thay vi Chi-square, va tinh tu TOAN BO nhom, khong phai mau nho)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_g = len(x)
    if n_g <= 2:
        return None
    mean_xy = np.array([x.mean(), y.mean()])
    cov_xy = np.cov(x, y, ddof=1)
    c2 = 2 * (n_g - 1) / (n_g * (n_g - 2)) * stats.f.ppf(0.95, 2, n_g - 2)
    L = np.linalg.cholesky(cov_xy)
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ell = (L @ circle * np.sqrt(c2)).T + mean_xy
    return ax.plot(ell[:, 0], ell[:, 1], **kwargs)


du_lieu_ve = du_lieu_sach.sample(n=min(1000, len(du_lieu_sach)), random_state=42)
du_lieu_ve["Trang_thai_no"] = du_lieu_ve[bien_nhom].map({0: "Tra dung han (0)", 1: "Vo no (1)"})

pairplot_fig = sns.pairplot(
    du_lieu_ve, vars=bien_tai_chinh, hue="Trang_thai_no",
    palette="Set2", diag_kind="kde", plot_kws={"alpha": 0.4, "s": 20},
)
pairplot_fig.fig.suptitle(
    "Bieu do phan tan cap (Pairplot) theo nhom", y=1.02, fontsize=12, weight="bold"
)
# (SUA: bo elip long trong tung o Pairplot - ly do xem comment ham
#  draw_mean_confidence_ellipse ben tren. Elip duoc tach thanh hinh rieng ben duoi.)

plt.savefig("pairplot_phan_nhom.png", dpi=300)
plt.close()
print("-> Da luu: pairplot_phan_nhom.png")


# (SUA: BO Boxplot - de bai (muc 4.2 template) chi yeu cau dung 3 hinh:
#  Heatmap, Pairplot, Elip tin cay 95%. Boxplot khong nam trong yeu cau,
#  bo di de "ve chinh xac theo de yeu cau", tranh nop du thua khong can thiet.)


# 3. Elip tin cay 95% cho VECTOR TRUNG BINH - hinh rieng, dung TOAN BO du lieu
#    tung nhom (khong phai mau 1000 dong dung de ve Pairplot) de nhat quan
#    voi so lieu da dung trong kiem dinh Hotelling T^2 o Chuong 3.
bien_x_elip = "person_income"
bien_y_elip = "loan_percent_income"
idx_x = bien_tai_chinh.index(bien_x_elip)
idx_y = bien_tai_chinh.index(bien_y_elip)

fig, ax = plt.subplots(figsize=(7, 6))

for Xg, mau, nhan in [(nhom_0, "#4C72B0", "Tra dung han (0)"),
                       (nhom_1, "#DD8452", "Vo no (1)")]:
    x_full = Xg[:, idx_x]
    y_full = Xg[:, idx_y]

    # Tam (centroid) cua tung nhom - tinh tu toan bo nhom
    ax.scatter(x_full.mean(), y_full.mean(), s=160, marker="X", color=mau,
               edgecolor="black", linewidth=1.4, zorder=5, label=nhan)

    draw_mean_confidence_ellipse(x_full, y_full, ax, color=mau, lw=2.4,
                                  linestyle="-")

ax.set_xlabel("Thu nhap (person_income)")
ax.set_ylabel("Ty le no/thu nhap (loan_percent_income)")
ax.set_title("Elip tin cay 95% cho VECTOR TRUNG BINH\ntheo nhom Tra dung han / Vo no",
              fontsize=12, weight="bold")
ax.legend(loc="upper right")
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("elip_tin_cay_95.png", dpi=300)
plt.close()
print("-> Da luu: elip_tin_cay_95.png")

print("\nHoan tat toan bo qua trinh tinh toan va xuat hinh anh!")