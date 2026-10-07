### Overall (all folds, all horizons 1-48 h, origins every 6 h)

| model | MAE | RMSE | MAPE_% | n |
|---|---|---|---|---|
| LightGBM | 1,297.36 | 1,686.99 | 2.46 | 68352 |
| Seasonal naive (same hour last week) | 2,621.37 | 3,961.53 | 5.00 | 68352 |
| Prophet | 2,662.16 | 3,305.10 | 5.23 | 68352 |
| SARIMA(1,0,1)(1,1,1)24 | 4,069.82 | 5,513.76 | 7.95 | 68352 |
| Seasonal naive (same hour, latest day) | 5,153.60 | 7,156.63 | 9.93 | 68352 |

### MAE [MW] by horizon block

| model | 01-06h | 07-12h | 13-24h | 25-48h |
|---|---|---|---|---|
| LightGBM | 1,016 | 1,234 | 1,299 | 1,383 |
| Prophet | 2,665 | 2,665 | 2,661 | 2,661 |
| SARIMA(1,0,1)(1,1,1)24 | 1,722 | 2,898 | 3,689 | 5,140 |
| Seasonal naive (same hour last week) | 2,635 | 2,634 | 2,629 | 2,611 |
| Seasonal naive (same hour, latest day) | 3,903 | 3,926 | 3,966 | 6,367 |

### MAE [MW] by selected horizon

| model | h=1 | h=6 | h=12 | h=24 | h=36 | h=48 |
|---|---|---|---|---|---|---|
| LightGBM | 809 | 1,170 | 1,247 | 1,297 | 1,380 | 1,368 |
| Prophet | 2,644 | 2,876 | 2,877 | 2,873 | 2,873 | 2,878 |
| SARIMA(1,0,1)(1,1,1)24 | 589 | 2,663 | 3,498 | 3,990 | 5,551 | 5,751 |
| Seasonal naive (same hour last week) | 2,624 | 2,685 | 2,686 | 2,675 | 2,663 | 2,642 |
| Seasonal naive (same hour, latest day) | 3,982 | 4,169 | 4,208 | 4,252 | 6,850 | 6,884 |

### MAE [MW] by day type

| model | holiday/bridge/xmas | normal weekday | weekend |
|---|---|---|---|
| LightGBM | 1,497 | 1,300 | 1,258 |
| Prophet | 3,950 | 2,351 | 3,175 |
| SARIMA(1,0,1)(1,1,1)24 | 4,875 | 3,384 | 5,548 |
| Seasonal naive (same hour last week) | 7,837 | 2,503 | 2,010 |
| Seasonal naive (same hour, latest day) | 5,422 | 4,303 | 7,111 |

| daytype | rows |
|---|---|
| holiday/bridge/xmas | 3308 |
| normal weekday | 45652 |
| weekend | 19392 |

### 90 % prediction interval: raw quantile regression vs. conformalised

| horizon | coverage_raw_% | coverage_conformal_% | width_raw_MW | width_conformal_MW |
|---|---|---|---|---|
| 1 | 84.0 | 95.2 | 3,121.3 | 4,126.0 |
| 6 | 77.8 | 86.9 | 4,210.0 | 5,214.8 |
| 12 | 78.1 | 91.2 | 4,544.2 | 6,403.5 |
| 24 | 77.6 | 91.6 | 4,650.0 | 6,375.4 |
| 36 | 77.9 | 90.7 | 4,947.1 | 6,886.6 |
| 48 | 78.3 | 90.5 | 4,968.7 | 6,786.6 |

Folds: 4 x 91 days, ends 2025-12-31.
