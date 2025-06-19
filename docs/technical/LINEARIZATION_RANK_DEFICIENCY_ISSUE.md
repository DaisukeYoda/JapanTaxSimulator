# 線形化システムのランク不足問題

## 概要

DSGEモデルの線形化において、係数行列Aが深刻なランク不足（rank 5/27）を示しており、Klein法による解法が不安定になっています。

## 問題の詳細

### 1. 症状

```
Matrix A shape: (27, 27)
A matrix rank: 5
B matrix rank: 25
```

- 27×27の正方行列Aのランクが5しかない
- Blanchard-Kahn条件が満たされない
- 線形化が自動的に簡易版にフォールバック

### 2. 原因分析

#### 2.1 前方視的構造の限定性

モデルの27個の内生変数のうち、前方視的項（forward-looking terms）を持つのは5つのみ：

| 変数 | 説明 | 前方視的項 |
|------|------|------------|
| C | 消費 | C_{t+1} |
| Lambda | 限界効用 | Lambda_{t+1} |
| q | 実質為替レート | q_{t+1} |
| Rk_gross | 資本収益率 | Rk_{t+1} |
| pi_gross | インフレ率 | pi_{t+1} |

残り22変数は静的（現在と過去の変数のみで決定）。

#### 2.2 方程式の分類

前方視的な方程式（5個）：
- 方程式3: 消費オイラー方程式
- 方程式5: 異時点間最適化条件
- 方程式6: 国際資産条件
- 方程式8: 投資オイラー方程式
- 方程式14: ニューケインジアン・フィリップス曲線

静的な方程式（22個）：
- 生産関数、要素需要、政府予算制約など

### 3. 技術的詳細

#### 3.1 行列Aの構造

```python
# 特異値分解の結果
Singular values:
s[0] = 1.83e+00
s[1] = 1.41e+00
s[2] = 6.96e-01
s[3] = 3.58e-01
s[4] = 2.20e-01
s[5] = 1e-10以下（22個）
```

実効ランク = 5（数値誤差を除く）

#### 3.2 Klein法の前提違反

Klein (2000)法は、前方視的変数の数と爆発的固有値の数が一致することを前提としていますが：

- 前方視的変数: 5個
- 爆発的固有値: 不定（ランク不足のため）

## 影響

### 1. 研究への影響

- 厳密なDSGE理論に基づく線形化が使用できない
- 簡易線形化への自動フォールバックが発生
- Blanchard-Kahn条件の検証が不可能

### 2. 実用上の対処

現在のコードは自動的に簡易線形化にフォールバック：

```python
# 警告メッセージ
"⚠️ Klein linearization failed, falling back to simple method"
```

## 解決策の提案

### 1. 短期的対処（実装済み）

```python
# 明示的に簡易線形化を使用
simulator = EnhancedTaxSimulator(
    model,
    use_simple_linearization=True
)
```

### 2. 中期的改善案

#### 案A: Klein法の修正

前方視的/静的ブロックに分割して解く：

```python
def solve_klein_partitioned(A, B):
    # 前方視的ブロックの抽出
    forward_mask = np.linalg.norm(A, axis=1) > 1e-12
    A_forward = A[forward_mask, :]
    
    # 縮小システムで解く
    # ...
```

#### 案B: 代替解法の実装

- Sims (2002)のQZ分解法
- Uhlig (1999)の方法
- Christiano (2002)の射影法

### 3. 長期的改善案

#### モデル構造の見直し

1. **動的要素の追加**
   - 習慣形成の強化
   - 投資調整コストの動的化
   - 期待形成メカニズムの拡張

2. **変数の統合**
   - 静的な定義式の削除
   - 変数の集約化

## 再現手順

```python
# 1. モデルの読み込み
from src.dsge_model import DSGEModel, ModelParameters
params = ModelParameters.from_json('config/parameters.json')
model = DSGEModel(params)

# 2. 定常状態計算
steady_state = model.compute_steady_state()
model.steady_state = steady_state

# 3. 線形化の実行
from src.linearization_improved import ImprovedLinearizedDSGE
linearizer = ImprovedLinearizedDSGE(model, steady_state)
linear_system = linearizer.build_system_matrices()

# 4. ランクの確認
print(f"Rank of A: {np.linalg.matrix_rank(linear_system.A)}")
# Output: Rank of A: 5
```

## 参考文献

- Klein, P. (2000). "Using the generalized Schur form to solve a multivariate linear rational expectations model." Journal of Economic Dynamics and Control, 24(10), 1405-1423.
- Sims, C. A. (2002). "Solving linear rational expectations models." Computational Economics, 20(1-2), 1-20.
- Blanchard, O. J., & Kahn, C. M. (1980). "The solution of linear difference models under rational expectations." Econometrica, 48(5), 1305-1311.

## 更新履歴

- 2025-06-19: 初版作成（問題の発見と分析）
- 対処法: 簡易線形化への自動フォールバック実装済み