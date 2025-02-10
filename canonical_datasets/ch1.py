import pandas as pd

# 讀取 canonical_dataset_TAIR10.txt
input_file = "canonical_dataset_TAIR10.txt"
output_file = "filtered_chr1_dataset.txt"

# 使用 pandas 讀取資料
data = pd.read_csv(input_file, sep="\t", header=None)
data.columns = ["GeneID", "Expression", "Chromosome", "Strand", "Start", "End", "Exon_Starts", "Exon_Ends"]

# 篩選 Chromosome 欄位為 'chr1'
filtered_data = data[data["Chromosome"] == "chr1"]

# 將篩選後的資料輸出到檔案
filtered_data.to_csv(output_file, sep="\t", index=False, header=False)

print(f"篩選完成，僅保留 chr1 的資料，結果儲存到 {output_file} 中。")
