from MainUI import MainUI
from CNN import CNN
from KNN import KNN
import os
modeltype = "KNN"
if modeltype == "KNN":
    if os.path.exists("knn.sav"):
        pass
    else:
        print("Kindly wait for a few minutes............")
        knnobj = KNN(3)
        knnobj.skl_knn()
else:
    if os.path.exists("cnn.hdf5"):
        pass
    else:
        print("Kindly wait a few minutes.........")
        cnnobj = CNN()
        cnnobj.build_and_compile_model()
        cnnobj.train_and_evaluate_model()
        cnnobj.save_model()
MainUIobj = MainUI(modeltype)
MainUIobj.mainloop()
MainUIobj.cleanup()
