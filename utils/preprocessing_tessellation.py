import skmob
# create a TrajDataFrame from a list
data_list = [[1, 39.984094, 116.319236, '2008-10-23 13:53:05'], [1, 39.984198, 116.319322, '2008-10-23 13:53:06'], [1, 39.984224, 116.319402, '2008-10-23 13:53:11'], [1, 39.984211, 116.319389, '2008-10-23 13:53:16']]
tdf = skmob.TrajDataFrame(data_list, latitude=1, longitude=2, datetime=3)
# print a portion of the TrajDataFrame
print(tdf.head())