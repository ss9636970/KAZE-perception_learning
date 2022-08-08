import numpy as np
import cv2
import logging

def showpic(pic):              #顯示圖片
    cv2.imshow('RGB', pic)     #顯示 RGB 的圖片
    cv2.waitKey(0)             #有這段才不會有bug

def readpic(p):                #讀入圖片
    return cv2.imread(p)
    
def savepic(img, p):           #儲存圖片
    cv2.imwrite(p, img)
    
#讀取多張照片
def path2pic(img):
    pic = []
    for i in img:
        p = readpic(i)
        pic.append(p)
    return pic

def sumlist(l, n):
    c = 0
    for i in l:
        c += i
    return c / n

# feature extractor
# Feature extractor
def extract_features(image_path, vector_size=32):
    image = readpic(image_path)
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.KAZE_create()
        # Dinding image keypoints
        kps = alg.detect(image)
        if kps:
            pass
        else:
            return np.ones(vector_size * 64, dtype=np.float64)
        # Getting first 32 of them. 
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc

def images_features(paths, vector_sizes=32):
    N = paths.shape[0]
    outputs = extract_features(paths[0], vector_size=vector_sizes)[np.newaxis, :]
    for p in range(1, N):
        feature = extract_features(paths[p], vector_size=vector_sizes)[np.newaxis, :]
        outputs = np.concatenate([outputs, feature], axis=0)
    return outputs

def create_logger(path, log_file):
    # config
    logging.captureWarnings(True)     # 捕捉 py waring message
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    my_logger = logging.getLogger(log_file) #捕捉 py waring message
    my_logger.setLevel(logging.INFO)
    
    # file handler
    fileHandler = logging.FileHandler(path + log_file, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    my_logger.addHandler(fileHandler)
    
    # console handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)
    consoleHandler.setFormatter(formatter)
    my_logger.addHandler(consoleHandler)
    
    return my_logger


# top  n  %  accuracy
# both preds and truths are same shape m by n (m is number of predictions and n is number of classes)
def top_n_accuracy(preds_array, truths, n):
    N = preds_array.shape[0]
    best_n = np.argsort(preds_array, axis=1)[:, -n:]
    successes = 0
    for i in range(preds_array.shape[0]):
      if truths[i] in best_n[i, :]:
        successes += 1
    return float(successes / N)
