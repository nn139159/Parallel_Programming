# LBPH

## 簡介
本專案為使用 C++ 與 CUDA（可選）實作的臉部辨識相關演算法系統，支援多平台（x86、ARM）及多種編譯選項，包含序列版與 OpenMP 平行化版本，並可選擇編譯 CUDA 加速版本。

---

## 系統需求
- Linux 或 Windows (需有 GNU Make 支援)
- g++ 編譯器 (建議 7 以上版本)
- CUDA Toolkit（選用，啟用 CUDA 加速時）
- OpenMP 支援的編譯器
- CUDA GPU（選用）

---

## 專案檔案結構
.
├── Data_Processing.cpp
├── NearestNeighborCollector.cpp
├── LBPH_Serial.cpp
├── LBPH_OpenMP.cpp
├── LBPH_Cuda.cpp         # CUDA 版本源碼（啟用 CUDA 時編譯）
├── lbphCuda.cu           # CUDA kernel 實作
├── main.cpp              # 主執行入口
├── omp_compare.cpp       # OpenMP 測試程式
├── Makefile
└── README.md             # 本文件


## Makefile 主要說明
 - 自動偵測作業系統（Linux / Windows）與架構（x86 / ARM）
 - 使用 g++ 編譯 C++ 程式，並根據平台加入 SIMD 優化標誌（x86 加 -mavx2，ARM 加 -march=native）
 - 支援 OpenMP 平行化（-fopenmp）
 - CUDA 支援可透過 USE_CUDA=1 開關啟用
 - 產生兩組執行檔：main 和 omp_compare

## 編譯說明

### 預設編譯（不使用 CUDA）

make

會產生兩個可執行檔：

- main
- omp_compare

### 啟用 CUDA 加速

make USE_CUDA=1

此命令將使用 nvcc 編譯 CUDA 程式碼並連結 CUDA 執行庫。


## 執行範例

./main
./omp_compare

## 清理編譯檔案

make clean

會刪除所有目標檔案與執行檔。



## 執行參數說明

程式接受最多三個命令列參數：

1. `database`（整數，選擇資料集）
    - `1`：使用 Extended Yale B+ 資料集
    - 其他值（預設）：使用 AT&T 臉部資料集

2. `max_samples`（整數，可選）
    - 指定最大樣本數量，用於限制讀取資料集中的樣本數

3. `train_ratio`（浮點數，可選，範圍 (0, 1)）
    - 訓練資料比例（例如 `0.9` 表示 90% 資料用於訓練，10% 用於測試）
    - 輸入若不在 0 到 1 之間，程式會輸出錯誤並終止

範例執行：

```bash
./main 1 1000 0.9

表示使用 Extended Yale B+ 資料集，最大樣本數 1000，訓練比例 90%。


## 資料集路徑說明
依據 database 參數，程式會載入對應資料集路徑：

database == 1 時，路徑為：

../../extendedyaleb_cropped_full

其他情況，路徑為：

../../att_faces

此專案內預設支援上述兩個資料集。若要使用其他資料集，請自行實作資料讀取與預處理邏輯，並調整程式對應路徑及資料格式。
數據集可以在以下連結下載
AT&T Database of Faces:
https://www.kaggle.com/datasets/kasikrit/att-database-of-faces
Extended Yale B (Cropped, Full):
https://www.kaggle.com/datasets/jensdhondt/extendedyaleb-cropped-full

