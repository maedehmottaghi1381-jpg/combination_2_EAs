"""
Hybrid Ant Colony + Artificial Bee Colony (AC-ABC)
for Feature Selection on Bank Marketing Dataset (UCI id=222)

Important optimizations for large dataset:
- Sample subset for fitness evaluation (speed)
- F1-score instead of accuracy (class imbalance)
- Single train/test split for final eval
- 3-fold CV inside fitness (speed)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import clone


# ============================================================
# 1. Load Dataset
# ============================================================
print("=" * 60)
print("Loading Bank Marketing dataset (UCI id=222)")
print("=" * 60)

bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets

print(f"Original features shape: {X.shape}")
print(f"Targets shape: {y.shape}")

# Convert target to binary (yes=1, no=0)
y_binary = (y.iloc[:, 0].astype(str).str.strip() == 'yes').astype(int).values
print(f"Class distribution: {np.bincount(y_binary)}")

# ============================================================
# 2. Preprocessing: Encode categorical, scale numeric
# ============================================================
# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Impute missing values
if numeric_cols:
    num_imputer = SimpleImputer(strategy='median')
    X_num = num_imputer.fit_transform(X[numeric_cols])
else:
    X_num = np.empty((len(X), 0))

if cat_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_cat = cat_imputer.fit_transform(X[cat_cols])
else:
    X_cat = np.empty((len(X), 0))

# One-Hot Encode categorical features
from sklearn.preprocessing import OneHotEncoder
if X_cat.shape[1] > 0:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X_cat)
else:
    X_cat_encoded = np.empty((len(X), 0))

# Scale numeric features
if X_num.shape[1] > 0:
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
else:
    X_num_scaled = np.empty((len(X), 0))

# Combine
X_processed = np.hstack([X_num_scaled, X_cat_encoded]) if (X_num_scaled.size + X_cat_encoded.size) > 0 else X_num_scaled
print(f"Processed features shape: {X_processed.shape}")

# ============================================================
# 3. Train/Test Split
# ============================================================
# Subsample for speed (use 20% of data - still 9000 samples)
SAMPLE_SIZE = 8000
np.random.seed(42)
idx = np.random.choice(len(X_processed), SAMPLE_SIZE, replace=False)
X_sub = X_processed[idx]
y_sub = y_binary[idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Train class distribution: {np.bincount(y_train)}")

# Further subsample training set for fitness evaluation (SPEED)
FITNESS_SAMPLE = 2000
if len(X_train) > FITNESS_SAMPLE:
    fit_idx = np.random.choice(len(X_train), FITNESS_SAMPLE, replace=False)
    X_fit_sample = X_train[fit_idx]
    y_fit_sample = y_train[fit_idx]
else:
    X_fit_sample = X_train
    y_fit_sample = y_train

print(f"Fitness evaluation sample: {X_fit_sample.shape}")


# ============================================================
# 4. Fitness Function (using F1-score + small sample)
# ============================================================
def fitness_function(selected_features, X_s, y_s, alpha=0.99, beta=0.01):
    """
    Fitness = F1-score (3-fold CV) - penalty * feature ratio.
    Uses small sample for speed.
    """
    if np.sum(selected_features) == 0:
        return -1.0
    try:
        X_sel = X_s[:, selected_features]
        clf = SVC(kernel='rbf', C=1.0, random_state=42)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        f1_scores = []
        for tr_idx, val_idx in skf.split(X_sel, y_s):
            clf_clone = clone(clf)
            clf_clone.fit(X_sel[tr_idx], y_s[tr_idx])
            pred = clf_clone.predict(X_sel[val_idx])
            f1_scores.append(f1_score(y_s[val_idx], pred, zero_division=0))
        score = np.mean(f1_scores)
    except Exception:
        return -1.0

    n_sel = np.sum(selected_features)
    n_tot = X_s.shape[1]
    return alpha * score - beta * (n_sel / n_tot)


# ============================================================
# 5. ACO - Ant Colony Optimization
# ============================================================
def ant_colony_optimization(X_s, y_s, n_ants=10, max_iter=20,
                            alpha_pher=1.0, beta_heur=1.0,
                            rho=0.1, Q=1.0):
    n_features = X_s.shape[1]
    pheromone = np.ones(n_features) * 0.5

    # Heuristic: point-biserial correlation
    correlations = np.array([
        abs(np.corrcoef(X_s[:, j], y_s)[0, 1]) if np.std(X_s[:, j]) > 1e-10 else 0.0
        for j in range(n_features)
    ])
    correlations = np.nan_to_num(correlations, nan=0.0)
    heuristic = correlations / (correlations.max() + 1e-10)

    best_mask = np.zeros(n_features, dtype=bool)
    best_fitness = -1.0
    convergence = []

    for t in range(max_iter):
        ant_solutions = []
        ant_fits = []

        for _ in range(n_ants):
            probs = (pheromone ** alpha_pher) * ((heuristic + 0.01) ** beta_heur)
            probs = probs / probs.sum()
            # Cap number of selected features (max 30% of total)
            max_sel = max(1, int(n_features * 0.3))
            mask = np.zeros(n_features, dtype=bool)
            selected = np.random.choice(n_features, size=max_sel, replace=False, p=probs)
            mask[selected] = True

            fit = fitness_function(mask, X_s, y_s)
            ant_solutions.append(mask)
            ant_fits.append(fit)

        ant_fits = np.array(ant_fits)
        local_best = np.argmax(ant_fits)
        if ant_fits[local_best] > best_fitness:
            best_fitness = ant_fits[local_best]
            best_mask = ant_solutions[local_best].copy()

        pheromone *= (1 - rho)
        for sol, fit in zip(ant_solutions, ant_fits):
            if fit > 0:
                pheromone[sol] += Q * fit
        pheromone = np.clip(pheromone, 0.01, 10.0)
        convergence.append(best_fitness)

    return best_mask, best_fitness, convergence


# ============================================================
# 6. ABC - Artificial Bee Colony
# ============================================================
def artificial_bee_colony(X_s, y_s, n_bees=10, max_iter=20, limit=5):
    n_features = X_s.shape[1]
    n_employed = n_bees // 2

    # Initialize food sources with sparse binary
    max_sel = max(1, int(n_features * 0.3))
    foods = np.zeros((n_employed, n_features), dtype=bool)
    for i in range(n_employed):
        sel = np.random.choice(n_features, size=max_sel, replace=False)
        foods[i, sel] = True

    fitnesses = np.array([fitness_function(f, X_s, y_s) for f in foods])
    trials = np.zeros(n_employed, dtype=int)

    best_idx = np.argmax(fitnesses)
    best_mask = foods[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    convergence = [best_fitness]

    for t in range(max_iter):
        # Employed bees
        for i in range(n_employed):
            neighbor = foods[i].copy()
            # Flip 1-3 random bits
            n_flips = np.random.randint(1, min(4, n_features))
            flip_pos = np.random.choice(n_features, n_flips, replace=False)
            neighbor[flip_pos] = ~neighbor[flip_pos]
            if neighbor.sum() == 0:
                neighbor[np.random.randint(n_features)] = True
            # Cap max features
            if neighbor.sum() > max_sel:
                true_idx = np.where(neighbor)[0]
                drop = np.random.choice(true_idx, neighbor.sum() - max_sel, replace=False)
                neighbor[drop] = False

            new_fit = fitness_function(neighbor, X_s, y_s)
            if new_fit > fitnesses[i]:
                foods[i] = neighbor
                fitnesses[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout bees
        for i in range(n_employed):
            if trials[i] > limit:
                sel = np.random.choice(n_features, size=max_sel, replace=False)
                foods[i] = np.zeros(n_features, dtype=bool)
                foods[i, sel] = True
                fitnesses[i] = fitness_function(foods[i], X_s, y_s)
                trials[i] = 0

        cur_best = np.argmax(fitnesses)
        if fitnesses[cur_best] > best_fitness:
            best_fitness = fitnesses[cur_best]
            best_mask = foods[cur_best].copy()
        convergence.append(best_fitness)

    return best_mask, best_fitness, convergence


# ============================================================
# 7. Hybrid AC-ABC
# ============================================================
def ac_abc_hybrid(X_s, y_s, n_ants=5, n_bees=5, max_iter=20, migration_interval=5):
    n_features = X_s.shape[1]
    max_sel = max(1, int(n_features * 0.3))

    # ACO state
    pheromone = np.ones(n_features) * 0.5
    correlations = np.array([
        abs(np.corrcoef(X_s[:, j], y_s)[0, 1]) if np.std(X_s[:, j]) > 1e-10 else 0.0
        for j in range(n_features)
    ])
    correlations = np.nan_to_num(correlations, nan=0.0)
    heuristic = correlations / (correlations.max() + 1e-10)

    # ABC state
    n_employed = n_bees
    foods = np.zeros((n_employed, n_features), dtype=bool)
    for i in range(n_employed):
        sel = np.random.choice(n_features, size=max_sel, replace=False)
        foods[i, sel] = True
    food_fits = np.array([fitness_function(f, X_s, y_s) for f in foods])
    trials = np.zeros(n_employed, dtype=int)

    global_best_mask = foods[np.argmax(food_fits)].copy()
    global_best_fit = food_fits.max()
    convergence = [global_best_fit]

    print(f"\n  Initial global best F1: {global_best_fit:.4f}")

    for t in range(max_iter):
        # ===== ACO Phase =====
        ant_solutions, ant_fits = [], []
        for _ in range(n_ants):
            probs = (pheromone ** 1.0) * ((heuristic + 0.01) ** 1.0)
            probs = probs / probs.sum()
            mask = np.zeros(n_features, dtype=bool)
            sel = np.random.choice(n_features, size=max_sel, replace=False, p=probs)
            mask[sel] = True
            fit = fitness_function(mask, X_s, y_s)
            ant_solutions.append(mask)
            ant_fits.append(fit)

        ant_fits = np.array(ant_fits)
        pheromone *= (1 - 0.1)
        for sol, fit in zip(ant_solutions, ant_fits):
            if fit > 0:
                pheromone[sol] += 1.0 * fit
        pheromone = np.clip(pheromone, 0.01, 10.0)

        # ===== ABC Phase =====
        for i in range(n_employed):
            neighbor = foods[i].copy()
            n_flips = np.random.randint(1, min(4, n_features))
            flip_pos = np.random.choice(n_features, n_flips, replace=False)
            neighbor[flip_pos] = ~neighbor[flip_pos]
            if neighbor.sum() == 0:
                neighbor[np.random.randint(n_features)] = True
            if neighbor.sum() > max_sel:
                true_idx = np.where(neighbor)[0]
                drop = np.random.choice(true_idx, neighbor.sum() - max_sel, replace=False)
                neighbor[drop] = False

            new_fit = fitness_function(neighbor, X_s, y_s)
            if new_fit > food_fits[i]:
                foods[i] = neighbor
                food_fits[i] = new_fit
                trials[i] = 0
            else:
                trials[i] += 1

        for i in range(n_employed):
            if trials[i] > 5:
                sel = np.random.choice(n_features, size=max_sel, replace=False)
                foods[i] = np.zeros(n_features, dtype=bool)
                foods[i, sel] = True
                food_fits[i] = fitness_function(foods[i], X_s, y_s)
                trials[i] = 0

        # ===== Global best update =====
        aco_best = np.argmax(ant_fits)
        if ant_fits[aco_best] > global_best_fit:
            global_best_fit = ant_fits[aco_best]
            global_best_mask = ant_solutions[aco_best].copy()

        abc_best = np.argmax(food_fits)
        if food_fits[abc_best] > global_best_fit:
            global_best_fit = food_fits[abc_best]
            global_best_mask = foods[abc_best].copy()

        convergence.append(global_best_fit)

        # ===== Migration =====
        if (t + 1) % migration_interval == 0:
            worst_abc = np.argmin(food_fits)
            foods[worst_abc] = global_best_mask.copy()
            food_fits[worst_abc] = global_best_fit
            pheromone[global_best_mask] += 1.0 * global_best_fit
            pheromone = np.clip(pheromone, 0.01, 10.0)
            print(f"  [iter {t+1}] Migration | Best F1: {global_best_fit:.4f}")

    return global_best_mask, global_best_fit, convergence


# ============================================================
# 8. Final Evaluation (on full test set with F1 and Accuracy)
# ============================================================
def final_evaluate(selected_mask, X_train, y_train, X_test, y_test):
    if np.sum(selected_mask) == 0:
        return 0.0, 0.0, 0
    clf = SVC(kernel='rbf', C=1.0, random_state=42)
    clf.fit(X_train[:, selected_mask], y_train)
    y_pred = clf.predict(X_test[:, selected_mask])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    return acc, f1, int(np.sum(selected_mask))


# ============================================================
# 9. Run All
# ============================================================
print("\n" + "=" * 60)
print("Running algorithms (this may take a few minutes...)")
print("=" * 60)

print("\n[1/3] ACO...")
aco_mask, aco_fit, aco_conv = ant_colony_optimization(
    X_fit_sample, y_fit_sample, n_ants=10, max_iter=15
)
aco_acc, aco_f1, aco_n = final_evaluate(aco_mask, X_train, y_train, X_test, y_test)
print(f"ACO     - Acc: {aco_acc:.4f} | F1: {aco_f1:.4f} | Features: {aco_n}/{X_train.shape[1]}")

print("\n[2/3] ABC...")
abc_mask, abc_fit, abc_conv = artificial_bee_colony(
    X_fit_sample, y_fit_sample, n_bees=10, max_iter=15
)
abc_acc, abc_f1, abc_n = final_evaluate(abc_mask, X_train, y_train, X_test, y_test)
print(f"ABC     - Acc: {abc_acc:.4f} | F1: {abc_f1:.4f} | Features: {abc_n}/{X_train.shape[1]}")

print("\n[3/3] AC-ABC Hybrid...")
acabc_mask, acabc_fit, acabc_conv = ac_abc_hybrid(
    X_fit_sample, y_fit_sample, n_ants=5, n_bees=5, max_iter=15
)
acabc_acc, acabc_f1, acabc_n = final_evaluate(acabc_mask, X_train, y_train, X_test, y_test)
print(f"AC-ABC  - Acc: {acabc_acc:.4f} | F1: {acabc_f1:.4f} | Features: {acabc_n}/{X_train.shape[1]}")


# ============================================================
# 10. Final Comparison
# ============================================================
print("\n" + "=" * 60)
print("Final Comparison")
print("=" * 60)

results = pd.DataFrame({
    'Algorithm': ['ACO (Ant)', 'ABC (Bee)', 'AC-ABC (Hybrid)'],
    'Accuracy': [aco_acc, abc_acc, acabc_acc],
    'F1-Score': [aco_f1, abc_f1, acabc_f1],
    'Selected Features': [aco_n, abc_n, acabc_n],
    'Fitness (CV F1)': [aco_fit, abc_fit, acabc_fit]
}).sort_values('F1-Score', ascending=False).reset_index(drop=True)

results.index = results.index + 1
print("\n" + results.to_string())
print("\nDone!")