# mgcreate.py
import numpy as np

def Plop(data, VH, back=0):
    ans = np.zeros(VH, float) + back

    vmax, hmax = VH
    V, H = data.shape

    vctr, hctr = V // 2, H // 2
    vactr, hactr = vmax // 2, hmax // 2

    valo = vactr - vctr
    if valo < 0:
        valo = 0

    vahi = vactr + vctr
    if vahi >= vmax:
        vahi = vmax

    halo = hactr - hctr
    if halo < 0:
        halo = 0

    hahi = hactr + hctr
    if hahi >= hmax:
        hahi = hmax

    vblo = vctr - vactr
    if vblo <= 0:
        vblo = 0

    vbhi = vctr + vactr
    if vbhi >= V:
        vbhi = V

    hblo = hctr - hactr
    if hblo <= 0:
        hblo = 0

    hbhi = hctr + hactr
    if hbhi >= H:
        hbhi = H

    if vahi - valo != vbhi - vblo:
        vbhi = vblo + vahi - valo

    if hahi - halo != hbhi - hblo:
        hbhi = hblo + hahi - halo

    ans[valo:vahi, halo:hahi] = data[vblo:vbhi, hblo:hbhi] + 0

    return ans