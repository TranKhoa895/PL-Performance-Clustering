import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

output_folder = 'outputs'


'''DATA COLLECTION'''
# file_path = "fbref_data.html"

# all_tables = pd.read_html(file_path)
# squad_stats = all_tables[0]

# squad_stats.columns = ['_'.join(col).strip() for col in squad_stats.columns.values]
# squad_stats.to_csv("squad_stats_pl.csv", index = False)


'''DATA PREPROCESSING'''
df = pd.read_csv('squad_stats_pl.csv')

df.columns = [
    'Squad', 'Players', 'Age', 'Poss', 'MP', 'Starts', 'Min', '90s',
    'Goals', 'Assists', 'G+A', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR',
    'Gls_per90', 'Ast_per90', 'G+A_per90', 'G-PK_per90', 'G+A-PK_per90'
]

print(df.head())

# Bổ sung thêm các chỉ số
df['Min_per_NonPK_Goal'] = df['Min'] / df['G-PK']   # Số phút cần để ghi 1 bàn thắng (không tính PK)
df['Poss_Efficiency'] = df['Goals'] / df['Poss']    # Tỷ lệ chuyển hóa kiểm soát bóng thành bàn thắng
df['Discipline_Score'] = df['CrdY'] + (df['CrdR'] * 3)  # Chỉ số kỷ luật (1 đỏ = 3 vàng)
print(df[['Squad', 'Min_per_NonPK_Goal', 'Poss_Efficiency', 'Discipline_Score']].head())


'''EDA'''
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', 1000) 

# THỐNG KÊ MÔ TẢ
print('Thống kê mô tả:')
print(df.describe())

# Scatterplot cho thống kê mô tả
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Poss', y='Goals', s=100, hue='Goals', palette='coolwarm')

plt.axhline(df['Goals'].mean(), color='red', linestyle='--', alpha=0.5, label='Trung bình bàn thắng')
plt.axvline(df['Poss'].mean(), color='blue', linestyle='--', alpha=0.5, label='Trung bình kiểm soát bóng')

for i, row in df.iterrows():
    plt.text(row['Poss']+0.2, row['Goals']+0.2, row['Squad'], fontsize=9)

plt.title('Phân tích hiệu suất: Kiểm soát bóng vs Bàn thắng', fontsize=15)
plt.legend()
plt.savefig(os.path.join(output_folder,'performance_scatter.png'))
print("Đã tạo xong biểu đồ Scatterplot: performance_scatter.png")

# HỆ SỐ TƯƠNG QUAN
correlation = df[['Poss', 'Goals', 'Assists', 'Age', 'CrdY', 'CrdR']].corr()
print("Ma trận tương quan giữa các chỉ số chính:")
print(correlation['Goals'].sort_values(ascending=False))

# Heatmap cho mối tương quan 
plt.figure(figsize=(10,8))
sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0)
plt.title('Ma trận tương quan giữa các chỉ số')
plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
print('Đã tạo xong biểu đồ Heatmap: correlation_heatmap.png')


'''MODELING'''
# Lựa chọn các đặc trưng
cluster_features = ['Goals', 'Poss', 'Poss_Efficiency', 'Discipline_Score']
X = df[cluster_features]

# Chuẩn hóa dữ liệu 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tìm số cụm tối ưu bằng Elbow Method 
wcss = []
for i in range (1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Vẽ biểu đồ Elbow
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', color='black')
plt.title('Phương pháp Elbow để chọn số cụm K')
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('WCSS (Độ biến thiên nội bộ cụm)')
plt.savefig(os.path.join(output_folder, 'elbow_method.png'))
print("Đã tạo xong biểu đồ Elbow: elbow_method.png")

# Phân cụm với K=4
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Tính các chỉ số đánh giá
# Sihouette
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"Chỉ số Silhouette (Độ tách biệt cụm): {sil_score:.4f}")
# DB Index
db_index = davies_bouldin_score(X_scaled, df['Cluster'])
print(f"Davies-Bouldin Index (Độ tương tụ trung bình cụm): {db_index:.4f}")
# Calinski-Harabasz Index
ch_index = calinski_harabasz_score(X_scaled, df['Cluster'])
print(f"Calinski-Harabasz Index (Độ phân tán giữa các cụm và phân tán nội bộ cụm): {ch_index:.4f}")

# Tính giá trị trung bình 
mean_poss = df['Poss'].mean()
mean_goals = df['Goals'].mean()

# Trực quan kết quả phân cụm 
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Poss', y='Goals', hue='Cluster', 
                palette='viridis', s=200, style='Cluster', edgecolor='black')

plt.axvline(mean_poss, color='red', linestyle='--', alpha=0.5, label='TB Kiểm soát')
plt.axhline(mean_goals, color='blue', linestyle='--', alpha=0.5, label='TB Bàn thắng')

for i, row in df.iterrows():
    plt.text(row['Poss']+0.3, row['Goals']+0.3, row['Squad'], fontsize=10, fontweight='bold')

plt.title('PHÂN CỤM PHONG CÁCH THI ĐẤU PREMIER LEAGUE', fontsize=16)
plt.xlabel('Tỷ lệ kiểm soát bóng (%)')
plt.ylabel('Tổng số bàn thắng')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig(os.path.join(output_folder, 'team_clusters_final.png'))
print("Đã tạo xong biểu đồ phân cụm: team_clusters_final.png")

# Tính giá trị trung bình của các chỉ số cho mỗi cụm
cluster_analysis = df.groupby('Cluster')[['Goals', 'Poss', 'Poss_Efficiency', 'Discipline_Score']].mean()
print("Giá trị trung bình theo từng cụm")
print(cluster_analysis)

# Danh sách các đội theo từng cụm 
print("Danh sách đội bóng theo cụm")
for i in range(4):
    teams = df[df['Cluster'] == i]['Squad'].tolist()
    print(f"Cụm {i}: {', '.join(teams)}")


with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
    f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PHÂN CỤM\n")
    f.write("="*30 + "\n")
    f.write(f"Silhouette Score: {sil_score:.4f}\n")
    f.write(f"Davies-Bouldin Index: {db_index:.4f}\n")
    f.write(f"Calinski-Harabasz Index: {ch_index:.4f}\n")
    f.write("="*30 + "\n")
    f.write("Đã hoàn thành phân tích 20 đội bóng Premier League.")