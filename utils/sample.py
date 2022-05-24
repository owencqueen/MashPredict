import pandas as pd


def EPS_sample_mask(df, col, topq = 0.75, botq = 0.25):

    C = df.loc[:,col]
    print(C)
    print(C.quantile(q=topq))
    tquant_mask = (C > C.quantile(q=topq)).to_numpy()
    bquant_mask = (C < C.quantile(q=botq)).to_numpy()
    mask = tquant_mask & bquant_mask
    print('mask', sum(mask))

    return mask, tquant_mask, bquant_mask