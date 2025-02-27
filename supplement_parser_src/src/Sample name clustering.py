import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE

# Sample words (your dataset)
words = ["WT1","WT2","WT3","WT4", "Prn1_1","Prn1_2","Prn1_3","Prn1_4", "WT_H2O2_1", "WT_H2O2_2", "WT_H2O2_3", "WT_H2O2_4", "Prn1_H2O2_1","Prn1_H2O2_2","Prn1_H2O2_3","Prn1_H2O2_4",
         "SHS40_S1-C9_2113.d", "SHS40_S1-D9_2114.d", "SHS40_S1-A9_2111.d", "CHS40_S1-B9_2112.d", "CHS40_S1-E9_2115.d", "SHS40_S1-A10_2117.d", "CHS40_S1-B10_2118.d", "CHS40_S1-C10_2119.d", "CHS40_S1-D10_2120.d", "SHS40_S1-E10_2121.d", "CTN40_S1-A11_2123.d", "STN40_S1-B11_2124.d", "CTN40_S1-C11_2125.d", "STN40_S1-D11_2126.d", "STN40_S1-E11_2127.d", "CTN40_S1-A12_2129.d", "STN40_S1-B12_2130.d", "122_63_STN40_S1-C12_2131.d" ,"CTN40_S1-D12_2132.d", "CTN40_S1-E12_2133.d", "CHS36_S1-A7_2099.d", "CHS36_S1-B7_2100.d", "CHS36_S1-C7_2101.d", "CHS36_S1-D7_2102.d", "CHS36_S1-E7_2103.d", "CTN36_S1-A8_2105.d", "CTN36_S1-B8_2106.d", "CTN36_S1-C8_2107.d", "CTN36_S1-D8_2108.d", "CTN36_S1-E8_2109.d",
         "24-126_CC_1.d", "24-126_CC_2.d", "24-126_CC_3.d", "24-126_CC_4.d", "24-126_CP_1.d", "24-126_CP_2.d", "24-126_CP_3.d", "24-126_CP_4.d", "24-126_TC_1.d","24-126_TC_2.d","24-126_TC_3.d","24-126_TC_4.d", "24-126_TP_1.d", "24-126_TP_2.d", "24-126_TP_3.d", "24-126_TP_4.d"]


# Create a character frequency vector for each word
def char_frequency(word):
    freq = {}
    for char in word:
        freq[char] = freq.get(char, 0) + 1
    return freq

# Create DataFrame of character frequencies
vector_df = pd.DataFrame([char_frequency(word) for word in words]).fillna(0)

# Calculate Pearson distance matrix
distance_matrix = pd.DataFrame(index=words, columns=words)

for i, word1 in enumerate(words):
    for j, word2 in enumerate(words):
        if i < j:
            # Calculate Pearson correlation
            corr, _ = pearsonr(vector_df.iloc[i], vector_df.iloc[j])
            # Convert to distance (1 - correlation)
            distance = 1 - corr
            distance_matrix.loc[word1, word2] = distance
            distance_matrix.loc[word2, word1] = distance

distance_matrix.fillna(0, inplace=True)

distance_matrix = distance_matrix.astype(float)

condensed_matrix = ssd.squareform(distance_matrix)

linked = sch.linkage(condensed_matrix, method='ward')

plt.figure(figsize=(10, 8))
sch.dendrogram(linked, labels=words, leaf_rotation=90)
plt.title("Hierarchical Clustering Dendrogram")
plt.savefig("Clustering_plot.tiff", format="tiff", dpi=300, bbox_inches="tight")
plt.show()

max_distance = 0.5  # Tune this threshold
clusters = fcluster(linked, max_distance, criterion='distance')

cluster_df = pd.DataFrame({"Word": words, "Cluster": clusters})
print(cluster_df)



#PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(vector_df)

cluster_df["PCA1"] = pca_result[:, 0]
cluster_df["PCA2"] = pca_result[:, 1]


explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(10, 8))
sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", palette="tab10", data=cluster_df, s=100, edgecolor="black")


for i, row in cluster_df.iterrows():
    plt.text(row["PCA1"], row["PCA2"], row["Word"], fontsize=8, alpha=0.7)


plt.title("PCA Projection of Clusters")
plt.xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}%)")
plt.ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}%)")


plt.legend(title="Cluster", bbox_to_anchor=(1, 1))
plt.savefig("PCA_plot.tiff", format="tiff", dpi=300, bbox_inches="tight")
plt.show()




# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(vector_df)

cluster_df["tSNE1"] = tsne_result[:, 0]
cluster_df["tSNE2"] = tsne_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x="tSNE1", y="tSNE2", hue="Cluster", palette="tab10", data=cluster_df, s=100, edgecolor="black")

for i, row in cluster_df.iterrows():
    plt.text(row["tSNE1"], row["tSNE2"], row["Word"], fontsize=8, alpha=0.7)

plt.title("t-SNE Projection of Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1, 1))
plt.savefig("tsne_plot.tiff", format="tiff", dpi=300, bbox_inches="tight")
plt.show()
