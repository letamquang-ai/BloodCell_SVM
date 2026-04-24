# BloodCell_SVM
Here are the steps to run the codes.

## Get BDDC dataset
Click [here](https://github.com/KietLe2504/Computer-Vison-Blood-Detection/tree/main/dataset/BDDC) to get the BDDC dataset and restructure it:
```bash
bddc
  \ann
  \img
```

## Processing data
Run these Python files for preprocessing and feature extraction. Note that the Plop function in ```mgcreate.py``` is available in ***Appendix C.2*** of the book [Image Operators: Image Processing in Python](https://github.com/Shegsdev/deep-learning-books/blob/master/5.%20Computer%20Vision%20(CV)%20Book/Image%20Operators%20Image%20Processing%20in%20Python-2019.pdf) by Dr. Jason M. Kinser:
```bash
python preprocess.py
python feature_extraction.py
```

## Training using SVM models
Now just train 4 SVM models and get results:
```bash
python svm_linear.py
python svm_poly.py
python svm_rad.py
python svm_sig.py
```
