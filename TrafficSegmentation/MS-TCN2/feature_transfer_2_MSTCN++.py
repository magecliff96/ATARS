import numpy as np
import os

# 定義輸入和輸出資料夾
input_folder = r'/home/magecliff/Traffic_Recognition/Carom_TempSeg/features/imgnet'
output_folder = r'/home/oort/MS-TCN2/data/carom/features'

# 確保輸出資料夾存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 對輸入資料夾中的所有 .npy 文件進行轉換
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        input_file = os.path.join(input_folder, filename)
        output_file = os.path.join(output_folder, filename)
        
        # 載入原始的 .npy 文件
        features = np.load(input_file)

        # 檢查原始數據的形狀是否符合 1x2048xTx1x1
        if features.shape[0] == 1 and features.shape[1] == 2048 and features.shape[3] == 1 and features.shape[4] == 1:
            # 將數據轉換為 2048xT 的形狀
            T = features.shape[2]
            reshaped_features = features.reshape(2048, T)

            # 保存為新的 .npy 文件
            np.save(output_file, reshaped_features)
            print(f'已成功將數據保存為 {output_file}')
        else:
            print(f'數據形狀不符合 1x2048xTx1x1，請檢查輸入文件：{input_file}')