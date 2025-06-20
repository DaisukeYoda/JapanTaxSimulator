# 社会保障システムのモデル統合計画

## 概要
JapanTaxSimulatorに社会保険料・年金システムを統合し、より包括的な財政分析を可能にする。

## 実装ロードマップ

### Phase 1: Quick Win（1-2週間）
**目標**: 最小限の変更で社会保険料の影響を反映

```python
# parameters.jsonに追加
"social_security": {
    "tau_ss_total": 0.30,  # 労使合計の社会保険料率
    "employee_share": 0.50,  # 労働者負担割合
    "pension_share": 0.60,   # 年金の割合
    "health_share": 0.35,    # 医療の割合
}

# 実装方法
# 現在：tau_l = 0.20 (所得税のみ)
# 改善：tau_l_total = tau_l + tau_ss * employee_share
#      = 0.20 + 0.30 * 0.50 = 0.35 (実効税率)
```

**利点**:
- 現在のモデル構造を維持
- すぐに実装可能
- 社会保険料の影響を近似的に捉える

### Phase 2: 社会保障勘定の明示化（1-3ヶ月）

**新規クラスの追加**:
```python
class SocialSecuritySystem:
    """社会保障システムの収支を明示的にモデル化"""
    
    def __init__(self, params):
        self.pension = PensionSystem(params)
        self.health = HealthInsystem(params)
        self.care = LongTermCareSystem(params)
        
    def compute_balance(self, state):
        # 保険料収入
        revenue = self.collect_premiums(state.wage, state.employment)
        # 給付支出
        expenditure = self.pay_benefits(state.retirees, state.patients)
        # 収支
        return revenue - expenditure
```

**DSGEモデルへの統合**:
- 政府部門を「一般政府」と「社会保障」に分離
- 財政ルールを拡張（年金積立金の運用等）

### Phase 3: 世代別分析（3-6ヶ月）

**簡易OLGモデル**:
```python
class SimpleOLG:
    """3世代（若年・中年・高齢）モデル"""
    
    generations = {
        "young": {"age": 20-39, "saves": True, "retired": False},
        "middle": {"age": 40-64, "saves": True, "retired": False},
        "old": {"age": 65+, "saves": False, "retired": True}
    }
    
    def lifecycle_optimization(self, generation):
        # 世代別の最適化問題
        if generation == "young":
            # 年金を考慮した貯蓄決定
            return optimize_with_pension_expectation()
```

### Phase 4: 完全統合モデル（6-12ヶ月）

**目標**: 人口動態を内生化した完全なモデル

```python
class FullyIntegratedModel:
    """税制・社会保障・人口動態の統合モデル"""
    
    def __init__(self):
        self.tax_system = TaxSystem()
        self.social_security = SocialSecuritySystem()
        self.demographics = DemographicProjection()
        
    def long_term_projection(self, horizon=50):
        # 50年間の財政持続可能性分析
        for t in range(horizon):
            # 人口構造の更新
            self.demographics.update()
            # 社会保障収支の計算
            ss_balance = self.social_security.compute_balance()
            # 一般政府収支との統合
            total_balance = self.integrate_budgets()
```

## 技術的課題と解決策

### 1. モデルの複雑性管理
- **課題**: 変数数の増加（30→50以上）
- **解決**: モジュラー設計で管理

### 2. 計算負荷
- **課題**: 世代別計算で計算量増大
- **解決**: 並列計算、JAX等の活用

### 3. データ取得
- **課題**: 詳細な社会保障データ
- **解決**: 厚労省統計、社会保障審議会資料

## 政策分析の応用例

### 1. 年金改革シミュレーション
```python
reforms = {
    "支給開始年齢": [65, 67, 70],
    "所得代替率": [50%, 45%, 40%],
    "保険料率": [18.3%, 20%, 22%]
}
```

### 2. 医療費抑制政策
- 自己負担率の変更
- ジェネリック推進
- 予防医療投資

### 3. 統合改革
- 消費税と社会保険料のバランス
- 世代間負担の公平性
- 持続可能性の確保

## 期待される成果

1. **より正確な財政分析**
   - 税と社会保障の一体的把握
   - 真の国民負担率の計算

2. **世代間公平性の評価**
   - 各世代の生涯負担と受益
   - 改革の世代別影響

3. **長期持続可能性**
   - 人口減少下での制度設計
   - 最適な負担と給付の組み合わせ

## 結論

社会保障の統合は技術的に可能かつ必要。段階的アプローチにより、短期的には簡易実装で効果を出しつつ、長期的には包括的なモデルへ発展させる。
