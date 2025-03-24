from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from causallearn.utils.cit import chisq
from causallearn.search.ConstraintBased.PC import pc_alg
import networkx as nx
import matplotlib.pyplot as plt


file_path = r"C:\Users\user\Desktop\산재보험\2차\df_cleaned.csv"

# CSV 파일 불러오기
df = pd.read_csv(file_path, encoding='utf-8-sig')  # 한글 포함 시 utf-8-sig 권장

drop_cols=[
    "A005A", "A005A03", "A005A04", "A005A05", "A005A06",
    "A005B01", "A005B02", "A005B03", "A005B04", "A005B05", "A005B06",
    "A005C01", "A005C02", "A005C03", "A005C04", "A005C05", "A005C06",
    "A005D01", "A005D02", "A005D03", "A005D04", "A005D05", "A005D06",
    "A005E01", "A005E02", "A005E03", "A005E04", "A005E05", "A005E06",
    "A005F01", "A005F02", "A005F03", "A005F04", "A005F05", "A005F06",
    "A005G01", "A005G02", "A005G03", "A005G04", "A005G05", "A005G06",
    "A005H01", "A005H02", "A005H03", "A005H04", "A005H05", "A005H06",
    "A005I01", "A005I02", "A005I03", "A005I04", "A005I05", "A005I06",
    "A005J01", "A005J02", "A005J03", "A005J04", "A005J05", "A005J06",
    "A005K01", "A005K02", "A005K03", "A005K04", "A005K05", "A005K06",
    "A005L01", "A005L02", "A005L03", "A005L04", "A005L05", "A005L06",
    "A005M01", "A005M02", "A005M03", "A005M04", "A005M05", "A005M06",
    "A005N01", "A005N02", "A005N03", "A005N04", "A005N05", "A005N06",
    "A005O01", "A005O02", "A005O03", "A005O04", "A005O05", "A005O06"
]


df = df.drop(columns=drop_cols, errors='ignore')
# object 타입 열들 삭제
object_cols_to_drop = df.select_dtypes(include='object').columns
df = df.drop(columns=object_cols_to_drop)

#현재 일자리 여부 판단 열 제거(contamination)
d_cols = [col for col in df.columns if col.startswith('D')]
df = df.drop(columns=d_cols)
#마찬가지
e_cols = [col for col in df.columns if col.startswith('E')]
df = df.drop(columns=e_cols)

g_cols = [col for col in df.columns if col.startswith('G013')or col.startswith('G012004')]
df = df.drop(columns=g_cols)


h_cols = [col for col in df.columns if col.startswith('H')]
df = df.drop(columns=h_cols)

i_cols = [col for col in df.columns if col.startswith('I')]
df = df.drop(columns=i_cols)




# === 2.5 결측률 80% 이상인 열 제거 ===
missing_thresh = 0.8
missing_ratio = df.isnull().mean()
high_missing_cols = missing_ratio[missing_ratio > missing_thresh].index.tolist()
print(f"🧹 결측률 80% 이상으로 제거된 열 수: {len(high_missing_cols)}")
df = df.drop(columns=high_missing_cols)

# === 2.6 분산이 거의 없는 열 제거 ===
low_variance_cols = []
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].nunique() <= 1:
        low_variance_cols.append(col)
    elif df[col].dtype in ["float32", "float64"] and df[col].var() < 1e-5:
        low_variance_cols.append(col)

print(f"🧹 분산 거의 없음으로 제거된 열 수: {len(low_variance_cols)}")
df = df.drop(columns=low_variance_cols)
y_col = 'first_yo'
X = df.drop(columns=[y_col,"second_yo"])
y = df[y_col]
# === 3. 소민 스타일 전처리 ===
X_encoded = X.copy()
for col in X_encoded.columns:
    if X_encoded[col].dtype in ["float64", "float32"] and X_encoded[col].nunique() > 20:
        X_encoded[col] = X_encoded[col].fillna(X_encoded[col].mean())
    else:
        if X_encoded[col].isnull().any():
            X_encoded[col] = X_encoded[col].fillna(f"question_{col}")
        X_encoded[col] = X_encoded[col].astype(str)
        X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col])

row_thresh = 0.9
X_encoded = X_encoded[X_encoded.isnull().mean(axis=1) < row_thresh]
# === 4. y값 처리 ===
if y.dtype == 'bool':
    y = y.astype(int)

from sklearn.feature_selection import SelectKBest, mutual_info_classif

# feature selection 대상
X_only = X_encoded
# 1. mutual_info 기반 상위 50개 뽑기
selector = SelectKBest(mutual_info_classif, k=30)
X_selected = selector.fit_transform(X_only, y)

# 2. 선택된 열 이름으로 DataFrame 복원
selected_cols = X_only.columns[selector.get_support()]
X_reduced = pd.DataFrame(X_selected, columns=selected_cols)

# 3. y 다시 붙이기
X_reduced[y_col] = y.reset_index(drop=True)

# 4. 구조 학습
cg = pc_alg(
    data=X_reduced.values,
    alpha=0.05,
    indep_test=chisq,
    stable=True,
    max_k=2,
    node_names=X_reduced.columns.tolist(),
    uc_rule=0,         # 기본적인 (default) rule
    uc_priority=0      # 우선순위 설정 (default: 0)
)
# 시각화
GraphUtils.plot_graph(cg, labels=X_reduced.columns.tolist())
plt.savefig(r"C:\Users\user\Desktop\산재보험\causal_graph.png", dpi=300, bbox_inches='tight')