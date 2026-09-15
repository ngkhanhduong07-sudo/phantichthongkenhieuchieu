import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

# ============================================================
# CHUONG 3: TIEN XU LY DU LIEU VA LAP TRINH TINH TOAN
# 3.1 -> 3.2 -> 3.3
# ============================================================


# ============================================================
# 1. DOC DU LIEU
# ============================================================

du_lieu = pd.read_csv("credit_risk_dataset.csv")


# 5 bien tai chinh dung trong phan tich
bien_tai_chinh = ["person_income", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

# Bien phan nhom
bien_nhom = "loan_status"

# Muc y nghia
muc_y_nghia = 0.01

# Chi giu cac cot can thiet
du_lieu_phan_tich = du_lieu[bien_tai_chinh + [bien_nhom]].copy()

# Loai cac dong bi thieu du lieu
du_lieu_phan_tich = du_lieu_phan_tich.dropna()

print("So quan sat sau khi loai du lieu thieu:", len(du_lieu_phan_tich))


# ============================================================
# 3.1. LAM SACH DU LIEU VA XU LY NGOAI LAI
#       BANG KHOANG CACH MAHALANOBIS
# ============================================================

print("\n============================================")
print("3.1. XU LY NGOAI LAI BANG MAHALANOBIS")
print("============================================")

X = du_lieu_phan_tich[bien_tai_chinh].to_numpy()

# Vector trung binh cua toan bo mau
vector_trung_binh = X.mean(axis=0)

# Ma tran hiep phuong sai mau
S = np.cov(X, rowvar=False, ddof=1)

# Ma tran nghich dao cua S
S_nghich_dao = np.linalg.inv(S)

# Khoang cach Mahalanobis binh phuong:
hieu = X - vector_trung_binh
mahalanobis_binh_phuong = np.einsum("ij,jk,ik->i", hieu, S_nghich_dao, hieu)

# Nguong toi han theo phan phoi Chi-binh-phuong
p = len(bien_tai_chinh)
nguong_mahalanobis = chi2.ppf(1 - muc_y_nghia, df=p)

# Xac dinh cac quan sat ngoai lai
la_ngoai_lai = mahalanobis_binh_phuong > nguong_mahalanobis
so_ngoai_lai = la_ngoai_lai.sum()

print("Nguong Mahalanobis D^2:", nguong_mahalanobis)
print("So quan sat bi loai:", so_ngoai_lai)

# Loai cac quan sat ngoai lai
du_lieu_sach = du_lieu_phan_tich[~la_ngoai_lai].copy()

print("So quan sat con lai sau khi loai ngoai lai:", len(du_lieu_sach))

# Luu du lieu sau khi lam sach (hỗ trợ tiếng Việt UTF-8)
ten_file_dulieu_sach = "credit_risk_dataset_sach.csv"
du_lieu_sach.to_csv(ten_file_dulieu_sach, index=False, encoding='utf-8-sig')
print("Da luu du lieu sau khi lam sach vao:", ten_file_dulieu_sach)


# ============================================================
# 3.2. KIEM TRA CAC GIA DINH THONG KE
# ============================================================

print("\n============================================")
print("3.2. KIEM TRA CAC GIA DINH THONG KE")
print("============================================")


# ------------------------------------------------------------
# 3.2.1. KIEM DINH CHUAN NHIEU CHIEU - MARDIA
# ------------------------------------------------------------

print("\n--- 3.2.1. KIEM DINH MARDIA ---")

X_sach_day_du = du_lieu_sach[bien_tai_chinh].to_numpy()

so_mau_mardia = min(5000, len(X_sach_day_du))
rng = np.random.default_rng(42)

if len(X_sach_day_du) > so_mau_mardia:
    chi_so = rng.choice(len(X_sach_day_du), size=so_mau_mardia, replace=False)
    X_sach = X_sach_day_du[chi_so]
else:
    X_sach = X_sach_day_du

n = len(X_sach)
p = len(bien_tai_chinh)

print("So quan sat dung cho Mardia =", n)

# Trung tam hoa du lieu
X_trung_tam = X_sach - X_sach.mean(axis=0)

# Ma tran hiep phuong sai mau
S_sach = np.cov(X_sach, rowvar=False, ddof=1)
S_sach_nghich_dao = np.linalg.inv(S_sach)

# Ma tran khoang cach Mahalanobis giua cac cap quan sat
Q = (X_trung_tam @ S_sach_nghich_dao @ X_trung_tam.T)
D2 = np.diag(Q)

# ----- Mardia Skewness -----
b1p = np.sum(Q ** 3) / (n ** 2)

# ----- Mardia Kurtosis -----
b2p = np.sum(D2 ** 2) / n

# Thong ke chuan hoa cua Mardia Skewness
skew_stat = n * b1p / 6
df_skew = p * (p + 1) * (p + 2) / 6
p_value_skew = stats.chi2.sf(skew_stat, df_skew)

# Thong ke Mardia Kurtosis
expected_kurtosis = p * (p + 2)
variance_kurtosis = (8 * p * (p + 2) / n)
z_kurt = (b2p - expected_kurtosis) / np.sqrt(variance_kurtosis)
p_value_kurt = 2 * stats.norm.sf(abs(z_kurt))

print("Mardia Skewness statistic =", skew_stat)
print("p-value Skewness =", p_value_skew)
print("Mardia Kurtosis statistic =", b2p)
print("Z Kurtosis =", z_kurt)
print("p-value Kurtosis =", p_value_kurt)

if p_value_skew < muc_y_nghia or p_value_kurt < muc_y_nghia:
    print("Ket luan: Bac bo H0 -> du lieu khong phu hop voi phan phoi chuan nhieu chieu.")
else:
    print("Ket luan: Chua du bang chung bac bo H0 -> chua co bang chung du lieu khong chuan nhieu chieu.")


# ------------------------------------------------------------
# 3.2.2. KIEM DINH DONG NHAT MA TRAN HIEP PHUONG SAI - BOX'S M
# ------------------------------------------------------------

print("\n--- 3.2.2. KIEM DINH BOX'S M ---")

nhom_0 = du_lieu_sach[du_lieu_sach[bien_nhom] == 0][bien_tai_chinh].to_numpy()
nhom_1 = du_lieu_sach[du_lieu_sach[bien_nhom] == 1][bien_tai_chinh].to_numpy()

n_0 = len(nhom_0)
n_1 = len(nhom_1)

S_0 = np.cov(nhom_0, rowvar=False, ddof=1)
S_1 = np.cov(nhom_1, rowvar=False, ddof=1)

# Ma tran hiep phuong sai gop
S_p = ((n_0 - 1) * S_0 + (n_1 - 1) * S_1) / (n_0 + n_1 - 2)

# Box's M
M = ((n_0 + n_1 - 2) * np.log(np.linalg.det(S_p)) - (n_0 - 1) * np.log(np.linalg.det(S_0)) - (n_1 - 1) * np.log(np.linalg.det(S_1)))

g = 2
sum_inv = (1 / (n_0 - 1) + 1 / (n_1 - 1) - 1 / (n_0 + n_1 - 2))
C = ((2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))) * sum_inv
F_box = M * (1 - C)
df1_box = (g - 1) * p * (p + 1) / 2
df2_box = ((p * (p + 1) / 2) * (1 / (g - 1) - C))
p_value_box = stats.f.sf(F_box, df1_box, df2_box)

print("Box's M =", M)
print("F xap xi =", F_box)
print("p-value =", p_value_box)
print("Bac tu do =", (df1_box, df2_box))

if p_value_box < muc_y_nghia:
    print("Ket luan: Bac bo H0 -> ma tran hiep phuong sai khong dong nhat.")
else:
    print("Ket luan: Chua du bang chung bac bo H0 -> co the chap nhan gia dinh dong nhat ma tran hiep phuong sai.")


# ============================================================
# 3.3. LAP TRINH TINH TOAN KIEM DINH HOTELLING T^2
# ============================================================

print("\n============================================")
print("3.3. KIEM DINH HOTELLING T^2")
print("============================================")

N = n_0 + n_1
vector_trung_binh_0 = nhom_0.mean(axis=0)
vector_trung_binh_1 = nhom_1.mean(axis=0)
hieu_vector_trung_binh = (vector_trung_binh_0 - vector_trung_binh_1)

print("\nVector trung binh nhom 0:")
print(vector_trung_binh_0)
print("\nVector trung binh nhom 1:")
print(vector_trung_binh_1)

# Ma tran hiep phuong sai gop S_p
print("\nMa tran hiep phuong sai gop S_p:")
print(S_p)

S_p_nghich_dao = np.linalg.inv(S_p)

# Hotelling T^2
T2 = ((n_0 * n_1) / N) * (hieu_vector_trung_binh @ S_p_nghich_dao @ hieu_vector_trung_binh)
print("\nHotelling T^2 =", T2)

# Quy doi T^2 sang F
F_thuc_te = ((N - p - 1) / (p * (N - 2))) * T2
bac_tu_do_1 = p
bac_tu_do_2 = N - p - 1
p_value = stats.f.sf(F_thuc_te, bac_tu_do_1, bac_tu_do_2)

print("F =", F_thuc_te)
print("Bac tu do F =", (bac_tu_do_1, bac_tu_do_2))
print("p-value =", p_value)


# ============================================================
# KET LUAN CUOI CUNG
# ============================================================

print("\n============================================")
print("KET LUAN KIEM DINH HOTELLING T^2")
print("============================================")

if p_value < muc_y_nghia:
    print("Bac bo H0 o muc y nghia 1%.")
    print("Vector trung binh cua 5 bien tai chinh giua hai nhom khac nhau co y nghia thong ke.")
else:
    print("Chua du bang chung bac bo H0 o muc y nghia 1%.")


# ============================================================
# ============================================================
# CHƯƠNG 4: TRỰC QUAN HÓA DỮ LIỆU ĐA BIẾN (VẼ BIỂU ĐỒ)
# ============================================================
print("\n============================================")
print("DANG TIEN HANH VE VA LUU BIEU DO CHO CHUONG 4...")
print("============================================")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 10})

# 1. Heatmap ma trận tương quan
corr_matrix = du_lieu_sach[bien_tai_chinh].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    linewidths=0.5,
    vmin=-1,
    vmax=1,
)
plt.title("Ma tran he so tuong quan Pearson (R)", fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig("heatmap_tuong_quan.png", dpi=300)
plt.close()
print("-> Da luu: heatmap_tuong_quan.png")


# 2. Pairplot kết hợp Elip tin cậy 95%
def draw_confidence_ellipse(x, y, ax, n_std=2.447, **kwargs):
  if x.size != y.size:
    return
  cov = np.cov(x, y)
  pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
  ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
  ellipse = Ellipse(
      (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs
  )
  scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
  transf = (
      plt.matplotlib.transforms.Affine2D()
      .rotate_deg(45)
      .scale(scale_x, scale_y)
      .translate(np.mean(x), np.mean(y))
  )
  ellipse.set_transform(transf + ax.transData)
  return ax.add_patch(ellipse)


du_lieu_ve = du_lieu_sach.sample(
    n=min(1000, len(du_lieu_sach)), random_state=42
)
du_lieu_ve["Trang_thai_no"] = du_lieu_ve[bien_nhom].map(
    {0: "Tra dung han (0)", 1: "Vo no (1)"}
)

pairplot_fig = sns.pairplot(
    du_lieu_ve,
    vars=bien_tai_chinh,
    hue="Trang_thai_no",
    palette="Set2",
    diag_kind="kde",
    plot_kws={"alpha": 0.4, "s": 20},
)
pairplot_fig.fig.suptitle(
    "Bieu do phan tan cap (Pairplot) kem Elip tin cay 95%",
    y=1.02,
    fontsize=12,
    weight="bold",
)

for i in range(len(bien_tai_chinh)):
  for j in range(len(bien_tai_chinh)):
    if i != j:
      ax = pairplot_fig.axes[i, j]
      x_0 = du_lieu_ve[du_lieu_ve[bien_nhom] == 0][bien_tai_chinh[j]]
      y_0 = du_lieu_ve[du_lieu_ve[bien_nhom] == 0][bien_tai_chinh[i]]
      x_1 = du_lieu_ve[du_lieu_ve[bien_nhom] == 1][bien_tai_chinh[j]]
      y_1 = du_lieu_ve[du_lieu_ve[bien_nhom] == 1][bien_tai_chinh[i]]
      if len(x_0) > 2:
        draw_confidence_ellipse(
            x_0,
            y_0,
            ax,
            edgecolor="#66c2a5",
            linestyle="--",
            linewidth=1.2,
            facecolor="none",
        )
      if len(x_1) > 2:
        draw_confidence_ellipse(
            x_1,
            y_1,
            ax,
            edgecolor="#fc8d62",
            linestyle="--",
            linewidth=1.2,
            facecolor="none",
        )

plt.savefig("pairplot_phan_nhom.png", dpi=300)
plt.close()
print("-> Da luu: pairplot_phan_nhom.png")


# 3. Boxplot so sánh phân phối
plt.figure(figsize=(14, 8))
for i, col in enumerate(bien_tai_chinh):
  plt.subplot(2, 3, i + 1)
  sns.boxplot(x=bien_nhom, y=col, data=du_lieu_sach, palette="Pastel1")
  plt.title(f"Phan phoi cua {col}", fontsize=10)
  plt.xlabel("Loan Status (0: Tra dung han, 1: Vo no)")
  plt.ylabel(col)

plt.tight_layout()
plt.savefig("boxplot_so_sanh_nhom.png", dpi=300)
plt.close()
print("-> Da luu: boxplot_so_sanh_nhom.png")

print("\nHoan tat toan bo qua trinh tinh toan va xuat hinh anh!")