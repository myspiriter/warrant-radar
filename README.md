# 個人版台股權證雷達 V6.9

- 雙軌評分：趨勢模型 + 事件模型
- 事件模型100分：事件衝擊20、洗盤品質20、止穩20、反彈確認20、事件風險20
- 新增「⚡事件型機會」頁籤
- 事件型高分股持續追蹤約5天，避免一次刷新就消失
- 前次高分追蹤改成 APP_DIR 持久檔案，不再使用 /tmp
- 資料信心、評分重點、雙軌分數缺值顯示「—」

## V6.9 ImportError 修正
補回 app.py 需要的權證函式：
- rank_warrants
- normalize_warrant_columns
- warrant_score
- moneyness_pct

保留 V6.9 雙軌評分與事件追蹤功能。
