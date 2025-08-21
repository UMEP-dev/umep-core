# Ready for python action!
import numpy as np

# import matplotlib.pylab as plt
# from numba import njit, types


def shadowingfunctionglobalradiation(a, azimuth, altitude, scale, forsvf):
    # %This m.file calculates shadows on a DEM
    # % conversion
    degrees = np.pi / 180.0
    # if azimuth == 0.0:
    # azimuth = 0.000000000001
    azimuth = np.dot(azimuth, degrees)
    altitude = np.dot(altitude, degrees)
    # % measure the size of the image
    sizex = a.shape[0]
    sizey = a.shape[1]
    if forsvf == 0:
        barstep = np.max([sizex, sizey])
        total = 100.0 / barstep  # dlg.progressBar.setRange(0, barstep)
    # % initialise parameters
    f = a
    dx = 0.0
    dy = 0.0
    dz = 0.0
    temp = np.zeros((sizex, sizey))
    index = 1.0
    # % other loop parameters
    amaxvalue = a.max()
    pibyfour = np.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1.0 / sinazimuth)
    dscos = np.abs(1.0 / cosazimuth)
    tanaltitudebyscale = np.tan(altitude) / scale
    # % main loop
    while amaxvalue >= dz and np.abs(dx) < sizex and np.abs(dy) < sizey:
        # while np.logical_and(np.logical_and(amaxvalue >= dz, np.abs(dx) <= sizex), np.abs(dy) <= sizey):(np.logical_and(amaxvalue >= dz, np.abs(dx) <= sizex), np.abs(dy) <= sizey):
        # if np.logical_or(np.logical_and(pibyfour <= azimuth, azimuth < threetimespibyfour), np.logical_and(fivetimespibyfour <= azimuth, azimuth < seventimespibyfour)):
        if (
            pibyfour <= azimuth
            and azimuth < threetimespibyfour
            or fivetimespibyfour <= azimuth
            and azimuth < seventimespibyfour
        ):
            dy = signsinazimuth * index
            dx = -1.0 * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos

        # % note: dx and dy represent absolute values while ds is an incremental value
        dz = ds * index * tanaltitudebyscale
        temp[0:sizex, 0:sizey] = 0.0
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = (dx + absdx) / 2.0 + 1.0
        xc2 = sizex + (dx - absdx) / 2.0
        yc1 = (dy + absdy) / 2.0 + 1.0
        yc2 = sizey + (dy - absdy) / 2.0
        xp1 = -((dx - absdx) / 2.0) + 1.0
        xp2 = sizex - (dx + absdx) / 2.0
        yp1 = -((dy - absdy) / 2.0) + 1.0
        yp2 = sizey - (dy + absdy) / 2.0
        temp[int(xp1) - 1 : int(xp2), int(yp1) - 1 : int(yp2)] = (
            a[int(xc1) - 1 : int(xc2), int(yc1) - 1 : int(yc2)] - dz
        )
        # f = np.maximum(f, temp)  # bad performance in python3. Replaced with fmax
        f = np.fmax(f, temp)
        index += 1.0

    f = f - a
    f = np.logical_not(f)
    sh = np.double(f)

    return sh


# @jit(nopython=True)
def shadowingfunction_20(a, vegdem, vegdem2, azimuth, altitude, scale, amaxvalue, bush, forsvf):
    # plt.ion()
    # fig = plt.figure(figsize=(24, 7))
    # plt.axis('image')
    # ax1 = plt.subplot(2, 3, 1)
    # ax2 = plt.subplot(2, 3, 2)
    # ax3 = plt.subplot(2, 3, 3)
    # ax4 = plt.subplot(2, 3, 4)
    # ax5 = plt.subplot(2, 3, 5)
    # ax6 = plt.subplot(2, 3, 6)
    # ax1.title.set_text('fabovea')
    # ax2.title.set_text('gabovea')
    # ax3.title.set_text('vegsh at ' + str(altitude))
    # ax4.title.set_text('lastfabovea')
    # ax5.title.set_text('lastgabovea')
    # ax6.title.set_text('vegdem')

    # This function casts shadows on buildings and vegetation units.
    # New capability to deal with pergolas 20210827

    # conversion
    degrees = np.pi / 180.0
    azimuth = azimuth * degrees
    altitude = altitude * degrees

    # measure the size of grid
    sizex = a.shape[0]
    sizey = a.shape[1]

    # progressbar for svf plugin
    if forsvf == 0:
        barstep = np.max([sizex, sizey])
        total = 100.0 / barstep
        # dlg.progressBar.setRange(0, barstep)
        # dlg.progressBar.setValue(0)

    # initialise parameters
    dx = 0.0
    dy = 0.0
    dz = 0.0
    temp = np.zeros((sizex, sizey), dtype=np.float32)
    tempvegdem = np.zeros((sizex, sizey), dtype=np.float32)
    tempvegdem2 = np.zeros((sizex, sizey), dtype=np.float32)
    templastfabovea = np.zeros((sizex, sizey), dtype=np.float32)
    templastgabovea = np.zeros((sizex, sizey), dtype=np.float32)
    bushplant = bush > 1.0
    sh = np.zeros((sizex, sizey), dtype=np.float32)  # shadows from buildings
    vbshvegsh = np.zeros((sizex, sizey), dtype=np.float32)  # vegetation blocking buildings
    vegsh = np.add(np.zeros((sizex, sizey), dtype=np.float32), bushplant, dtype=float)  # vegetation shadow
    f = a

    pibyfour = np.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1.0 / sinazimuth)
    dscos = np.abs(1.0 / cosazimuth)
    tanaltitudebyscale = np.tan(altitude) / scale
    # index = 1
    index = 0

    # new case with pergola (thin vertical layer of vegetation), August 2021
    dzprev = 0

    # main loop
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if (
            (pibyfour <= azimuth)
            and (azimuth < threetimespibyfour)
            or (fivetimespibyfour <= azimuth)
            and (azimuth < seventimespibyfour)
        ):
            dy = signsinazimuth * index
            dx = -1.0 * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos
        # note: dx and dy represent absolute values while ds is an incremental value
        dz = (ds * index) * tanaltitudebyscale
        tempvegdem[0:sizex, 0:sizey] = 0.0
        tempvegdem2[0:sizex, 0:sizey] = 0.0
        temp[0:sizex, 0:sizey] = 0.0
        templastfabovea[0:sizex, 0:sizey] = 0.0
        templastgabovea[0:sizex, 0:sizey] = 0.0
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int((dx + absdx) / 2.0)
        xc2 = int(sizex + (dx - absdx) / 2.0)
        yc1 = int((dy + absdy) / 2.0)
        yc2 = int(sizey + (dy - absdy) / 2.0)
        xp1 = int(-((dx - absdx) / 2.0))
        xp2 = int(sizex - (dx + absdx) / 2.0)
        yp1 = int(-((dy - absdy) / 2.0))
        yp2 = int(sizey - (dy + absdy) / 2.0)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = np.fmax(f, temp)  # Moving building shadow
        sh[(f > a)] = 1.0
        sh[(f <= a)] = 0.0
        fabovea = tempvegdem > a  # vegdem above DEM
        gabovea = tempvegdem2 > a  # vegdem2 above DEM

        # new pergola condition
        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        dzprev = dz
        vegsh2 = np.add(
            np.add(np.add(fabovea, gabovea, dtype=float), lastfabovea, dtype=float),
            lastgabovea,
            dtype=float,
        )
        vegsh2[vegsh2 == 4] = 0.0
        # vegsh2[vegsh2 == 1] = 0. # This one is the ultimate question...
        vegsh2[vegsh2 > 0] = 1.0

        vegsh = np.fmax(vegsh, vegsh2)
        vegsh[(vegsh * sh > 0.0)] = 0.0
        vbshvegsh = vegsh + vbshvegsh  # removing shadows 'behind' buildings

        # im1 = ax1.imshow(fabovea)
        # im2 = ax2.imshow(gabovea)
        # im3 = ax3.imshow(vegsh)
        # im4 = ax4.imshow(lastfabovea)
        # im5 = ax5.imshow(lastgabovea)
        # im6 = ax6.imshow(vegshtest)
        # im1 = ax1.imshow(tempvegdem)
        # im2 = ax2.imshow(tempvegdem2)
        # im3 = ax3.imshow(vegsh)
        # im4 = ax4.imshow(templastfabovea)
        # im5 = ax5.imshow(templastgabovea)
        # im6 = ax6.imshow(vegshtest)
        # plt.show()
        # plt.pause(0.05)

        index += 1.0

    sh = 1.0 - sh
    vbshvegsh[(vbshvegsh > 0.0)] = 1.0
    vbshvegsh = vbshvegsh - vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh

    # plt.close()
    # plt.ion()
    # fig = plt.figure(figsize=(24, 7))
    # plt.axis('image')
    # ax1 = plt.subplot(1, 3, 1)
    # im1 = ax1.imshow(vegsh)
    # plt.colorbar(im1)

    # ax2 = plt.subplot(1, 3, 2)
    # im2 = ax2.imshow(vegdem2)
    # plt.colorbar(im2)
    # plt.title('TDSM')

    # ax3 = plt.subplot(1, 3, 3)
    # im3 = ax3.imshow(vegdem)
    # plt.colorbar(im3)
    # plt.tight_layout()
    # plt.title('CDSM')
    # plt.show()
    # plt.pause(0.05)

    shadowresult = {"sh": sh, "vegsh": vegsh, "vbshvegsh": vbshvegsh}

    return shadowresult


# NOTE: Numba offers limited gains in this case
# @njit
def shadowingfunction_20_numba(
    a: np.ndarray,
    vegdem: np.ndarray,
    vegdem2: np.ndarray,
    azimuth: float,
    altitude: float,
    scale: float,
    amaxvalue: float,
    bush: np.ndarray,
    forsvf: int,
):  #  -> types.Tuple((types.float64[:, :], types.float64[:, :], types.float64[:, :])):
    # This function casts shadows on buildings and vegetation units.
    # New capability to deal with pergolas 20210827

    # conversion
    degrees = np.pi / 180.0
    azimuth = azimuth * degrees
    altitude = altitude * degrees

    # measure the size of grid
    sizex = a.shape[0]
    sizey = a.shape[1]

    # initialise parameters
    dx = 0.0
    dy = 0.0
    dz = 0.0
    temp = np.zeros((sizex, sizey))
    tempvegdem = np.zeros((sizex, sizey))
    tempvegdem2 = np.zeros((sizex, sizey))
    templastfabovea = np.zeros((sizex, sizey))
    templastgabovea = np.zeros((sizex, sizey))
    bushplant = bush > 1.0
    sh = np.zeros((sizex, sizey))  # shadows from buildings
    vbshvegsh = np.zeros((sizex, sizey))  # vegetation blocking buildings
    vegsh = np.zeros((sizex, sizey), dtype=np.float64)  # Initialize the array with zeros
    # Add bushplant values to the vegsh array
    for i in range(sizex):
        for j in range(sizey):
            vegsh[i, j] = bushplant[i, j]  # Assuming bushplant has the same shape
    f = a.copy()

    pibyfour = np.pi / 4.0
    threetimespibyfour = 3.0 * pibyfour
    fivetimespibyfour = 5.0 * pibyfour
    seventimespibyfour = 7.0 * pibyfour
    sinazimuth = np.sin(azimuth)
    cosazimuth = np.cos(azimuth)
    tanazimuth = np.tan(azimuth)
    signsinazimuth = np.sign(sinazimuth)
    signcosazimuth = np.sign(cosazimuth)
    dssin = np.abs(1.0 / sinazimuth) if sinazimuth != 0 else np.inf  # Avoid division by zero
    dscos = np.abs(1.0 / cosazimuth) if cosazimuth != 0 else np.inf  # Avoid division by zero
    tanaltitudebyscale = np.tan(altitude) / scale
    # index = 1
    index = 0

    # new case with pergola (thin vertical layer of vegetation), August 2021
    dzprev = 0

    # main loop
    while (amaxvalue >= dz) and (np.abs(dx) < sizex) and (np.abs(dy) < sizey):
        if (
            (pibyfour <= azimuth)
            and (azimuth < threetimespibyfour)
            or (fivetimespibyfour <= azimuth)
            and (azimuth < seventimespibyfour)
        ):
            dy = signsinazimuth * index
            dx = -1.0 * signcosazimuth * np.abs(np.round(index / tanazimuth))
            ds = dssin
        else:
            dy = signsinazimuth * np.abs(np.round(index * tanazimuth))
            dx = -1.0 * signcosazimuth * index
            ds = dscos
        # note: dx and dy represent absolute values while ds is an incremental value
        dz = (ds * index) * tanaltitudebyscale
        print(index, dz)
        tempvegdem[0:sizex, 0:sizey] = 0.0
        tempvegdem2[0:sizex, 0:sizey] = 0.0
        temp[0:sizex, 0:sizey] = 0.0
        templastfabovea[0:sizex, 0:sizey] = 0.0
        templastgabovea[0:sizex, 0:sizey] = 0.0
        absdx = np.abs(dx)
        absdy = np.abs(dy)
        xc1 = int((dx + absdx) / 2.0)
        xc2 = int(sizex + (dx - absdx) / 2.0)
        yc1 = int((dy + absdy) / 2.0)
        yc2 = int(sizey + (dy - absdy) / 2.0)
        xp1 = int(-((dx - absdx) / 2.0))
        xp2 = int(sizex - (dx + absdx) / 2.0)
        yp1 = int(-((dy - absdy) / 2.0))
        yp2 = int(sizey - (dy + absdy) / 2.0)

        tempvegdem[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dz
        tempvegdem2[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dz
        temp[xp1:xp2, yp1:yp2] = a[xc1:xc2, yc1:yc2] - dz

        f = np.fmax(f, temp)  # Moving building shadow
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f[i, j] > a[i, j]:
                    sh[i, j] = 1.0
                else:
                    sh[i, j] = 0.0
        fabovea = tempvegdem > a  # vegdem above DEM
        gabovea = tempvegdem2 > a  # vegdem2 above DEM

        # new pergola condition
        templastfabovea[xp1:xp2, yp1:yp2] = vegdem[xc1:xc2, yc1:yc2] - dzprev
        templastgabovea[xp1:xp2, yp1:yp2] = vegdem2[xc1:xc2, yc1:yc2] - dzprev
        lastfabovea = templastfabovea > a
        lastgabovea = templastgabovea > a
        dzprev = dz
        # Initialize vegsh2 with the same shape as the input arrays
        vegsh2 = np.zeros(fabovea.shape, dtype=np.float64)
        # Perform the additions directly
        vegsh2 += fabovea
        vegsh2 += gabovea
        vegsh2 += lastfabovea
        vegsh2 += lastgabovea

        for i in range(vegsh2.shape[0]):
            for j in range(vegsh2.shape[1]):
                if vegsh2[i, j] == 4:
                    vegsh2[i, j] = 0.0
                if vegsh2[i, j] > 0:
                    vegsh2[i, j] = 1.0

        vegsh = np.fmax(vegsh, vegsh2)
        for i in range(vegsh.shape[0]):
            for j in range(vegsh.shape[1]):
                if vegsh[i, j] * sh[i, j] > 0.0:
                    vegsh[i, j] = 0.0
        vbshvegsh = vegsh + vbshvegsh  # removing shadows 'behind' buildings

        index += 1.0

    sh = 1.0 - sh
    for i in range(vbshvegsh.shape[0]):
        for j in range(vbshvegsh.shape[1]):
            if vbshvegsh[i, j] > 0.0:
                vbshvegsh[i, j] = 1.0
    vbshvegsh = vbshvegsh - vegsh
    vegsh = 1.0 - vegsh
    vbshvegsh = 1.0 - vbshvegsh

    return sh, vegsh, vbshvegsh
