
bars_count = 30
long_bars_count = 30
compress_factor = 1.0
hist_idx = -bars_count
bar_size = 2
epsilon = 1.08
cnt = 1
for bar_idx in range(-bars_count, -bars_count-long_bars_count, -1):
    res_idx =
    print(("hist_idx=", hist_idx, ",bar_size=", bar_size, ",bar_idx=", bar_idx, ",cnt=", cnt, "res_idx="))
    compress_factor *= epsilon
    hist_idx -= bar_size
    bar_size = int(compress_factor * 2)
    cnt+=1
