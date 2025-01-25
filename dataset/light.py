def extract_first_sequence(fasta_file, output_file):
    with open(fasta_file, 'r') as file:
        output_lines = []
        is_first_sequence = False

        for line in file:
            if line.startswith(">"):  # 序列標題
                if is_first_sequence:  # 已擷取過第一筆資料，結束讀取
                    break
                is_first_sequence = True
                output_lines.append(line.strip())
            elif is_first_sequence:
                output_lines.append(line.strip())

    # 將擷取的第一筆資料寫入新檔案
    with open(output_file, 'w') as out_file:
        for line in output_lines:
            out_file.write(line + '\n')

    print(f"第一筆序列已擷取並儲存到 {output_file} 中。")

# 使用範例
input_fasta = "TAIR10.fa"   # 輸入的 .fa 檔案
output_fasta = "first_sequence.fa"  # 輸出的檔案
extract_first_sequence(input_fasta, output_fasta)
