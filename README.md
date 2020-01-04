# EE369
Machine Learning Medical 3D Voxel Classification

sttest文件夾中是經過散射變換的測試數據

mixup.py      ->數據增强
myRf.py       ->訓練隨即森林
output.txt    ->輸出分類結果
outputprob.txt->輸出分類概率
result.csv    ->輸出模板
rf.pkl        ->模型文件
st.py         ->散射變換
test.py       ->運行一鍵生成結果
train_val.csv ->原始label

運行test.py需要sttest文件夾以及 result.csv,rf.pkl,test.py
