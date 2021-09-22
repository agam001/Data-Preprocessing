import numpy as np 
import pandas as pd
from itertools import combinations as comb
from scipy.stats import chi2 

def chi2_test(df, columns):
  col_combo = comb(columns,2)

  for combo in col_combo:
    col1, col2 = combo
    col1_levels = df[col1].nunique() 
    col2_levels = df[col2].nunique() 

    cross_tab = pd.DataFrame()

    for x in df[col1].unique():
      value_count = df.query(f"{col1} == '{x}'")[col2].value_counts()
      cross_tab[x] = value_count
    
    row_sum = cross_tab.sum(axis=1)
    column_sum = cross_tab.sum(axis=0)
    grand_total = row_sum.sum()

    f_obs = cross_tab.values.flatten()
    f_obs = f_obs.astype(np.float64)

    f_exp = []
    for x in range(col1_levels):
      tmp = column_sum * row_sum[x]
      tmp = np.array(tmp,dtype=np.float64)
      f_exp.extend(tmp/grand_total)
    

    terms = (f_obs - f_exp)**2 / f_exp
    statistic = terms.sum()
    ddof = (col1_levels-1) * (col2_levels-1)

    p_value = 1-chi2.cdf(x= statistic, df = ddof)

    return cross_tab,statistic, p_value,terms, f_obs,f_exp


