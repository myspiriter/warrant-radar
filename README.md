# 個人版台股權證雷達

這是一個只給自己使用的 Streamlit Web App。每天開啟後，會把追蹤股票依「事件型反彈、趨勢回檔、突破、過熱」分類與排序，再把每檔股票的權證依成交量、剩餘天數、價內外、槓桿、Delta、價差與 IV 風險排序。

## 目前版本

- ✅ 今日標的排行
- ✅ 昨日訊號/績效紀錄（本機保存）
- ✅ NEXT 潛在標的
- ✅ 事件型反彈 / 趨勢回檔 / 突破 / 過熱分類
- ✅ 5萬＋5萬（或自訂資金）分批策略
- ✅ 權證硬篩選：成交量、到期天數、價內外
- ✅ 權證綜合分數
- ✅ 支援官方 TWSE 資料抓取
- ✅ 若權證完整條款抓不到，可在側欄上傳 CSV，App 會直接排序
- ✅ 評分權重可在側欄調整

## 安裝

1. 安裝 Python 3.11 或更新版本。
2. 在此資料夾開啟終端機：

```bash
pip install -r requirements.txt
streamlit run app.py
```

Windows 也可直接雙擊 `run_windows.bat`。
macOS 可在 Terminal 執行 `bash run_mac.sh`。

啟動後通常會自動開啟 `http://localhost:8501`。

## 手機開啟

電腦與手機在同一個 Wi‑Fi 時，Streamlit 啟動畫面會顯示 Network URL，例如：

`http://192.168.1.100:8501`

把這個網址輸入 iPhone Safari 即可。

## 官方資料來源

App 優先使用臺灣證券交易所公開端點：

- 每日上市股票行情：`https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX`
- 個股月日成交：`https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY`
- 上市權證每日成交 OpenAPI：`https://openapi.twse.com.tw/v1/opendata/t187ap42_L`

權證「履約價 / 到期日 / 行使比例 / Delta / IV / 買賣價差」若官方彙總表端點格式變動，App 會保留成交量資料，並提示你上傳券商匯出的 CSV 補齊欄位。

## 權證 CSV 欄位

建議欄位（中英文都可，App 會嘗試辨識）：

- warrant_code / 權證代號
- warrant_name / 權證名稱
- underlying / 標的代號
- issuer / 發行商
- warrant_type / 類型（CALL / PUT）
- price / 成交價
- volume / 成交量
- strike / 履約價
- expiry / 到期日
- bid / 委買
- ask / 委賣
- delta / Delta
- iv / IV
- effective_leverage / 有效槓桿

## 評分概念

### 現股分數（0–100）

- 趨勢 20
- 量價 20
- 技術位置 15
- 籌碼代理/動能 15
- 事件代理 15
- 基本面代理 15

本版不會假裝知道即時新聞語意；「事件分數」先用異常量價與跳空代理。正式要把新聞、法說、財報文字納入，可以再加新聞 API 或 AI 摘要。

### 權證分數（0–100）

- 成交量 20
- 剩餘天數 15
- 價內外 20
- Delta 15
- 有效槓桿 10
- 買賣價差 10
- IV 10

硬性淘汰預設：

- 成交量 < 300 張
- 剩餘天數 < 120 天
- 價外 > 15%

## 重要限制

這是決策輔助工具，不是自動下單系統。免費公開資料可能是盤後或延遲資料；若要做到盤中即時排行，應串接你券商提供的合法即時行情 API。
