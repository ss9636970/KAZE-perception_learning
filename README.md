# KAZE with perception learning

# 1.  training dataset
此任務為圖片分類任務，資料皆為狗的照片，並可以分為50種不同的狗，下方照片為舉例圖片

![n02111277_160](https://github.com/ss9636970/KAZE-perception_learning/blob/main/readme/n02111277_160.JPEG)![n02111277_160](https://github.com/ss9636970/KAZE-perception_learning/blob/main/readme/n02111500_113.jpg)



資料可分為:

63325張訓練資料

450張測試資料

450張驗證資料



# 2. Image feature extractor
本篇使用KAZE演算法提取圖片特徵，實作時以cv2為主要套件，該套件將每張圖片轉為2048維度的向量，

KAZE演算法計算圖片中像素之間的梯度值(gradient)作為圖片特徵，下圖為KAZE演算法計算特徵示意圖，

![KAZE](https://github.com/ss9636970/KAZE-perception_learning/blob/main/readme/KAZE.PNG)

# 3. Perception

本篇使用Perception作為圖片分類器，perception為線性分類器，可以視為單層的神經網路演算法，並在最後接上softmax激活函式。

本篇使用cross entropy loss為模型優化的目標函式的標準值，並用梯度下降法更新模型參數。



# 4. 程式碼說明

moduleClass.py 為Perception模型定義程式碼

funcion.py 為程式中運用到的函式，當中包括提取圖片特徵的函式，本篇使用cv2的KAZE_create函式提取圖片特徵

main.ipynb為執行模型訓練程式，當中包括讀取資料及特徵提取的執行程式碼





