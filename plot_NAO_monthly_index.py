import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("NAO_monthly_index.txt", sep=" ", skipinitialspace=True,
                 names=["year", "month", "index"])
df['datetime'] = pd.to_datetime(df.year.astype(str)+"/"+df.month.astype(str),format='%Y/%m')

df[df.year > 2004].plot('datetime', 'index')
plt.savefig("NAO_index_2004_onwards.png")

