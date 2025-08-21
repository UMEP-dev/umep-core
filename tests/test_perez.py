import numpy as np

from umep.util.SEBESOLWEIGCommonFiles.Perez_v3 import Perez_v3


def test_perez():
    radD = 200
    radI = 800
    jday = 100

    for zen_deg in range(0, 91, 5):
        for azi_deg in range(0, 361, 30):
            lv, _, _ = Perez_v3(zen_deg, azi_deg, radD, radI, jday, 1, 2)
            print("LV >>>", lv.min(), lv.mean(), lv.max())
            assert (~np.isnan(lv)).all()
