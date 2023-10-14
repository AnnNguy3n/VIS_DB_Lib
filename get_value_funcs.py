import numpy as np
import numba as nb


@nb.njit
def geomean(arr):
    log_sum = 0.0
    for num in arr:
        if num <= 0.0: return 0.0
        log_sum += np.log(num)

    return np.exp(log_sum/len(arr))


@nb.njit
def harmean(arr):
    deno = 0.0
    for num in arr:
        if num <= 0.0: return 0.0
        deno += 1.0/num

    return len(arr)/deno


@nb.njit
def get_inv_max_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest):
    '''
    Output: CtyMax, GeoMax, HarMax
    '''
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size-1)
    for i in range(size-1, 0, -1):
        start, end = INDEX[i], INDEX[i+1]
        temp = WEIGHT[start:end]
        max_ = np.max(temp)
        arr_idx_max = np.where(temp == max_)[0]
        if arr_idx_max.shape[0] == 1:
            arr_profit[i-1] = PROFIT[start:end][arr_idx_max[0]]
            if arr_profit[i-1] <= 0.0:
                break
        else:
            arr_profit[i-1] = interest

    GeoMax = geomean(arr_profit)
    HarMax = harmean(arr_profit)
    if GeoMax == 0.0:
        CtyMax = -1
    else:
        start, end = INDEX[0], INDEX[1]
        temp = WEIGHT[start:end]
        max_ = np.max(temp)
        arr_idx_max = np.where(temp == max_)[0] + start
        if arr_idx_max.shape[0] == 1:
            CtyMax = SYMBOL[arr_idx_max[0]]
        else:
            CtyMax = -1

    return CtyMax, GeoMax, HarMax


@nb.njit
def get_inv_ngn_infor(WEIGHT, INDEX, PROFIT, SYMBOL, interest):
    '''
    Output: Nguong, Top5Coms, GeoNgn, HarNgn
    '''
    size = INDEX.shape[0] - 1
    arr_profit = np.zeros(size-1)
    temp_profit = np.zeros(size-1)
    max_profit = -1.0

    list_loop = np.zeros((size-1)*5)
    for k in range(size-1, 0, -1):
        start, end = INDEX[k], INDEX[k+1]
        temp_weight = WEIGHT[start:end].copy()
        temp_weight[::-1].sort()
        list_loop[5*(k-1):5*k] = temp_weight[:5]

    list_loop = np.unique(list_loop)
    for v in list_loop:
        C = WEIGHT > v
        temp_profit[:] = 0.0
        for i in range(size-1, 0, -1):
            start, end = INDEX[i], INDEX[i+1]
            if np.count_nonzero(C[start:end]) == 0:
                temp_profit[i-1] = 1.06
            else:
                temp_profit[i-1] = PROFIT[start:end][C[start:end]].mean()

        new_profit = geomean(temp_profit)
        if new_profit > max_profit:
            Nguong = v
            max_profit = new_profit
            arr_profit[:] = temp_profit[:]

    HarNgn = harmean(arr_profit)

    start, end = INDEX[0], INDEX[1]
    mask = WEIGHT[start:end] > Nguong
    values = WEIGHT[start:end][mask]
    coms = SYMBOL[start:end][mask]
    mask_ = np.argsort(values)[::-1]
    if len(mask_) > 5:
        mask_ = mask_[:5]

    Top5Coms = coms[mask_]
    return Nguong, Top5Coms, max_profit, HarNgn
