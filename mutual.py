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


from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
import json

# === 1. 딕셔너리 불러오기 (영문 → 한글)
with open(r"C:\Users\user\Desktop\산재보험\variable_dict.json", "r", encoding='utf-8-sig') as f:
    variable_dict = json.load(f)
def to_kor(var): return variable_dict.get(var, var)

# === 2. 기존 변수 중요도 측정 (Mutual Info 기반)
mi_base = mutual_info_classif(X_encoded, y, discrete_features='auto', random_state=42)
mi_base_series = pd.Series(mi_base, index=X_encoded.columns).sort_values(ascending=False)

# === 3. 조합 생성
top_k = 12         # 상위 변수 몇 개로 조합할지
min_comb = 2       # 최소 변수 개수
max_comb = 4       # 최대 변수 개수

top_features = mi_base_series.head(top_k).index.tolist()
combined_features = []

for r in range(min_comb, max_comb + 1):
    combined_features += list(combinations(top_features, r))

print(f"🔍 생성된 복합 변수 조합 수: {len(combined_features)}")

# === 4. 복합 변수 생성
X_aug = X_encoded.copy()
new_feature_names = []

for comb in combined_features:
    col_name = "_x_".join(comb)
    X_aug[col_name] = X_encoded[list(comb)].prod(axis=1)
    new_feature_names.append(col_name)

# === 5. Mutual Info 계산
mi_all = mutual_info_classif(X_aug[new_feature_names], y, discrete_features='auto', random_state=42)
mi_result = pd.Series(mi_all, index=new_feature_names).sort_values(ascending=False)

# === 6. 한글 이름 매핑 후 결과 출력
print("\n✅ 상위 복합 변수 조합 (한글 매핑):")
for i, (feat, score) in enumerate(mi_result.head(15).items(), 1):
    kor_name = " × ".join([to_kor(x) for x in feat.split("_x_")])
    print(f"{i:2d}. {kor_name} → {score:.4f}")
