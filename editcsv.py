import pandas as pd

filereader = pd.read_csv('Data/Wimbledon-men-2013.csv',iterator='True',chunksize=10)

# for row in filereader:
#     # row = row[columns]
#     row[:8].to_csv('subtrain.csv',header=True,index=0)
#     row[8:].to_csv('subtest.csv',header=True,index=0)
#     break

for row in filereader:
    # row = row[columns]
    row[:8].to_csv('subtrain.csv', mode='a',header=False,index=0)
    row[8:].to_csv('subtest.csv', mode='a',header=False,index=0)
