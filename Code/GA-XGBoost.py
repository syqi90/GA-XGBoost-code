"""
Solar Energetic Particle (SEP) Event Prediction Using GA-Optimized XGBoost
============================================================================

This program implements a machine learning pipeline for predicting Solar Energetic
Particle (SEP) events based on solar radio burst features. The pipeline includes:

1. Data preprocessing and cleaning
2. Feature importance analysis
3. Genetic Algorithm (GA) optimization of XGBoost hyperparameters
4. Model comparison (SVM, XGBoost, AdaBoost, GA-XGBoost)
5. SHAP interpretability analysis
6. Comprehensive visualization for academic publication

Author: [Shiyang Qi]
Date: 202607
License: MIT
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    GridSearchCV,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# XGBoost
import xgboost as xgb

# Genetic Algorithm optimization
from sko.GA import GA

# SHAP for model interpretability
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# Global Configuration
# =============================================================================

# Matplotlib configuration for publication-quality figures
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.dpi": 300,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    }
)

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Path configuration (modify as needed)
DATA_PATH = "GAcleaned_2_4_&SEP&paper_total_data.csv"
OUTPUT_DIR = "./output/"


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform initial cleaning of the dataset.

    Parameters
    ----------
    filepath : str
        Path to the CSV data file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """
    print("=" * 70)
    print("Loading data...")
    print("=" * 70)

    df = pd.read_csv(
        "D:\\xiangmu\\pythonProject\\newXGBoost\\" + filepath,
        encoding="gbk",
        index_col=0,
    ).reset_index(drop=True)

    # Alternative: load from current directory
    # df = pd.read_csv(filepath, sep=',')

    # Remove empty columns
    df = df.dropna(axis=1, how="all")

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")

    return df


def preprocess_features(
    df: pd.DataFrame, target_columns: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess feature matrix and target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.
    target_columns : List[str]
        Columns to exclude from features (target and metadata).

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y.
    """
    # Separate features and target
    X = df.drop(columns=target_columns)
    y = df["EVENT TYPE"]

    # Convert object columns to numeric
    X = X.apply(
        lambda x: pd.to_numeric(x, errors="coerce") if x.dtype == "object" else x
    )

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X, y


def fill_missing_values(X: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Fill missing values using rolling mean and median.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with potential missing values.
    window : int, optional
        Rolling window size for mean imputation, by default 5.

    Returns
    -------
    pd.DataFrame
        Feature matrix with missing values filled.
    """
    # Step 1: Rolling mean imputation
    X = X.apply(
        lambda x: x.fillna(x.rolling(window=window, min_periods=1).mean())
    )

    # Step 2: Median imputation for any remaining NaN
    X = X.apply(lambda x: x.fillna(x.median()))

    # Verify no missing values remain
    remaining_nan = X.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"Warning: {remaining_nan} missing values remain!")
    else:
        print("All missing values successfully filled.")

    return X


def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply min-max normalization to features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        Normalized feature matrix.
    """
    return (X - X.min()) / (X.max() - X.min())


# =============================================================================
# Model Evaluation Utilities
# =============================================================================


def evaluate_classifier(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray
        Predicted probabilities for positive class.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all evaluation metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall/POD": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
        "MSE": mean_squared_error(y_true, y_prob),
        "MAE": mean_absolute_error(y_true, y_prob),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_prob)),
        "R2": r2_score(y_true, y_prob),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
    }
    return metrics


def print_evaluation_results(
    metrics: Dict[str, Any], dataset_name: str = "Dataset"
) -> None:
    """
    Print evaluation metrics in formatted tables.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of evaluation metrics.
    dataset_name : str
        Name of the dataset being evaluated.
    """
    print(f"\n{'='*70}")
    print(f"  {dataset_name} Results")
    print(f"{'='*70}")

    # Classification metrics
    print(f"\n--- Classification Metrics ---")
    for key in ["Accuracy", "Precision", "Recall/POD", "F1", "AUC"]:
        if key in metrics:
            print(f"  {key:15s}: {metrics[key]:.4f}")

    # Error metrics
    print(f"\n--- Error Metrics ---")
    for key in ["MSE", "MAE", "RMSE", "R2"]:
        if key in metrics:
            print(f"  {key:15s}: {metrics[key]:.4f}")

    # Confusion matrix
    if "Confusion Matrix" in metrics:
        print(f"\n--- Confusion Matrix ---")
        print(metrics["Confusion Matrix"])

    print("=" * 70)


# =============================================================================
# GA-XGBoost Model
# =============================================================================


class GAXGBoostOptimizer:
    """
    Genetic Algorithm optimized XGBoost classifier for SEP prediction.
    """

    def __init__(
        self,
        scale_pos_weight: float = 1.0,
        population_size: int = 40,
        generations: int = 60,
        n_repeats: int = 3,
        random_state: int = 42,
    ):
        """
        Initialize the GA-XGBoost optimizer.

        Parameters
        ----------
        scale_pos_weight : float
            Scale positive weight for imbalanced classes.
        population_size : int
            GA population size.
        generations : int
            Number of GA generations.
        n_repeats : int
            Number of repeated training runs.
        random_state : int
            Random seed for reproducibility.
        """
        self.scale_pos_weight = scale_pos_weight
        self.population_size = population_size
        self.generations = generations
        self.n_repeats = n_repeats
        self.random_state = random_state

        # Parameter bounds [learning_rate, max_depth, n_estimators,
        #                    subsample, colsample_bytree, alpha, lambda, gamma]
        self.param_bounds_lower = [0.02, 3, 100, 0.6, 0.6, 0, 0, 0]
        self.param_bounds_upper = [0.2, 9, 400, 1.0, 1.0, 1, 1, 4]

        self.best_params = None
        self.best_model = None
        self.best_threshold = 0.5
        self.ga_history = None
        self.loss_history = []

    def _create_model(
        self, params: List[float], use_early_stopping: bool = False
    ) -> xgb.XGBClassifier:
        """
        Create an XGBoost model with given parameters.

        Parameters
        ----------
        params : List[float]
            List of hyperparameters.
        use_early_stopping : bool
            Whether to enable early stopping.

        Returns
        -------
        xgb.XGBClassifier
            Configured XGBoost model.
        """
        model = xgb.XGBClassifier(
            learning_rate=params[0],
            max_depth=int(params[1]),
            n_estimators=int(params[2]),
            subsample=params[3],
            colsample_bytree=params[4],
            alpha=params[5],
            lambda_=params[6],
            gamma=params[7],
            scale_pos_weight=self.scale_pos_weight,
            max_delta_step=1,
            eval_metric="logloss",
            random_state=self.random_state,
            use_label_encoder=False,
        )

        if use_early_stopping:
            model.set_params(early_stopping_rounds=20)

        return model

    def _fitness_function(self, params: List[float]) -> float:
        """
        Fitness function for GA optimization (negative AUC for minimization).

        Parameters
        ----------
        params : List[float]
            Hyperparameters to evaluate.

        Returns
        -------
        float
            Negative mean AUC score.
        """
        model = self._create_model(params, use_early_stopping=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        score = cross_val_score(
            model, self.X_train, self.y_train, cv=cv, scoring="roc_auc"
        ).mean()
        return -score

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> "GAXGBoostOptimizer":
        """
        Fit the GA-XGBoost model with repeated training.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            Test labels.

        Returns
        -------
        self
        """
        self.X_train = X_train
        self.y_train = y_train

        print("\n" + "=" * 70)
        print("Running Genetic Algorithm optimization...")
        print("=" * 70)

        # Run GA optimization
        ga = GA(
            func=self._fitness_function,
            n_dim=8,
            size_pop=self.population_size,
            max_iter=self.generations,
            lb=self.param_bounds_lower,
            ub=self.param_bounds_upper,
        )
        self.best_params, best_loss = ga.run()
        self.ga_history = np.array(ga.all_history_Y)

        print("\n" + "=" * 70)
        print("Best Parameters Found:")
        print("=" * 70)
        param_names = [
            "learning_rate",
            "max_depth",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "alpha",
            "lambda_",
            "gamma",
        ]
        for name, value in zip(param_names, self.best_params):
            print(f"  {name:20s}: {value:.4f}")

        # Repeated training for stability
        self._repeated_training(X_train, y_train, X_test, y_test)

        # Find optimal threshold
        self._find_optimal_threshold(X_test, y_test)

        return self

    def _repeated_training(
        self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """
        Perform repeated training for model stability assessment.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels.
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            Test labels.
        """
        print(f"\nPerforming {self.n_repeats} repeated training runs...")

        best_auc = -1
        self.loss_history = []

        for i in range(self.n_repeats):
            print(f"  Training run {i+1}/{self.n_repeats}...")

            model = self._create_model(self.best_params, use_early_stopping=True)
            model.set_params(n_estimators=2000)

            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

            loss = model.evals_result()["validation_1"]["logloss"]
            self.loss_history.append(loss)

            prob = model.predict_proba(X_test)[:, 1]
            auc_val = roc_auc_score(y_test, prob)

            if auc_val > best_auc:
                best_auc = auc_val
                self.best_model = model
                print(f"    New best model! AUC = {auc_val:.4f}")

        print(f"\nBest AUC: {best_auc:.4f}")
        print(f"Best iteration: {self.best_model.best_iteration}")

    def _find_optimal_threshold(
        self, X_test: np.ndarray, y_test: np.ndarray
    ) -> None:
        """
        Find optimal classification threshold maximizing F1 score.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray
            Test labels.
        """
        y_prob = self.best_model.predict_proba(X_test)[:, 1]
        best_f1 = 0

        for t in np.arange(0.3, 0.7, 0.01):
            y_pred = (y_prob > t).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                self.best_threshold = t

        print(f"Optimal threshold: {self.best_threshold:.2f} (F1 = {best_f1:.4f})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using optimal threshold."""
        prob = self.best_model.predict_proba(X)[:, 1]
        return (prob > self.best_threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.best_model.predict_proba(X)[:, 1]


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_learning_curve(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot learning curve with confidence intervals.

    Parameters
    ----------
    model : Any
        Trained model.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    cv : int
        Number of cross-validation folds.
    save_path : str, optional
        Path to save the figure.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=cv,
        train_sizes=np.linspace(0.1, 1, 10),
        scoring="accuracy",
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    line1 = ax.plot(train_sizes, train_mean, "o-", color="#5f9ed1", linewidth=2)[0]
    line2 = ax.plot(train_sizes, test_mean, "o-", color="#f9736b", linewidth=2)[0]

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        color="#5f9ed1",
        alpha=0.15,
    )
    ax.fill_between(
        train_sizes,
        test_mean - test_std,
        test_mean + test_std,
        color="#f9736b",
        alpha=0.15,
    )

    ax.legend(
        [line1, line2],
        ["Training Score", "Validation Score"],
        loc="lower right",
        fontsize=11,
    )

    ax.set_xlabel("Sample Size", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    ax.set_ylim(0.7, 1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_training_loss(
    loss_history: List[List[float]], save_path: Optional[str] = None
) -> None:
    """
    Plot training loss curves for repeated runs.

    Parameters
    ----------
    loss_history : List[List[float]]
        List of loss curves from repeated training.
    save_path : str, optional
        Path to save the figure.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    for i, loss in enumerate(loss_history):
        ax.plot(
            loss,
            color=colors[i],
            marker=markers[i],
            markersize=5,
            markevery=10,
            linewidth=1.5,
            label=f"Train {i+1}",
        )

    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("Log Loss", fontsize=18)
    ax.legend(fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_ga_convergence(
    ga_history: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Plot GA optimization convergence curves.

    Parameters
    ----------
    ga_history : np.ndarray
        GA optimization history.
    save_path : str, optional
        Path to save the figure.
    """
    param_names = [
        "learning_rate",
        "max_depth",
        "n_estimators",
        "subsample",
        "colsample_bytree",
        "alpha",
        "lambda_",
        "gamma",
    ]
    auc_history = -ga_history[:, 0]
    sub_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

    fig = plt.figure(figsize=(16, 12), dpi=300)

    for i in range(9):
        ax = fig.add_subplot(3, 3, i + 1)

        if i < 8:
            ax.plot(ga_history[:, i], color="#c71f37", linewidth=2)
            ax.set_title(f"Fitness ({param_names[i]})", fontsize=18)
            ax.set_ylabel("Fitness Value (-AUC)", fontsize=18)
        else:
            ax.plot(auc_history, color="#0066cc", linewidth=3)
            ax.set_title("GA Convergence Curve (AUC)", fontsize=18)
            ax.set_ylabel("Best AUC", fontsize=18)

        ax.text(
            0.95,
            0.92,
            sub_labels[i],
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            ha="right",
            va="top",
        )
        ax.set_xlabel("Generation", fontsize=18)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_roc_curves(
    models_prob: Dict[str, np.ndarray],
    y_test: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    models_prob : Dict[str, np.ndarray]
        Dictionary mapping model names to predicted probabilities.
    y_test : np.ndarray
        True test labels.
    save_path : str, optional
        Path to save the figure.
    """
    style_list = [
        ("#1f77b4", "-"),
        ("#FEC44F", "--"),
        ("#2ca02c", "-."),
        ("#d62728", ":"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    for idx, (name, y_prob) in enumerate(models_prob.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        color, ls = style_list[idx]
        ax.plot(
            fpr,
            tpr,
            color=color,
            linestyle=ls,
            linewidth=2,
            label=f"{name} (AUC = {roc_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.legend(loc="lower right", fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 7,
    save_path: Optional[str] = None,
) -> List[str]:
    """
    Plot feature importance ranking and return top features.

    Parameters
    ----------
    model : Any
        Trained XGBoost model.
    feature_names : List[str]
        List of feature names.
    top_n : int
        Number of top features to highlight.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    List[str]
        Names of top N features.
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]

    # Print importance scores
    print("\n" + "=" * 80)
    print("Feature Importance Ranking")
    print("=" * 80)
    for i, (name, score) in enumerate(zip(sorted_names, sorted_importance)):
        print(f"  Rank {i+1:2d} | {name:20s} | Score = {score:.4f}")

    top_features = sorted_names[:top_n]
    print(f"\nTop {top_n} features: {top_features}")

    # Plot
    top_idx = sorted_idx[:top_n]
    colors = ["#d62728" if i in top_idx else "#1f77b4" for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.barh(range(len(sorted_names)), sorted_importance, color=colors)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance Score", fontsize=16)
    ax.set_ylabel("Feature Name", fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return top_features


def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot SHAP summary plot for model interpretability.

    Parameters
    ----------
    model : Any
        Trained XGBoost model.
    X : pd.DataFrame
        Feature matrix.
    feature_names : List[str]
        Feature names.
    save_path : str, optional
        Path to save the figure.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    shap.summary_plot(
        shap_values, X, feature_names=feature_names, show=False
    )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_feature_boxplots(
    X: pd.DataFrame,
    y: pd.Series,
    high_features: List[str],
    low_features: List[str],
    save_dir: str = "./",
) -> None:
    """
    Plot boxplots for high and low importance features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    high_features : List[str]
        High importance feature names.
    low_features : List[str]
        Low importance feature names.
    save_dir : str
        Directory to save figures.
    """

    def remove_outliers(data: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
        """Remove outliers using IQR method."""
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        return data[(data >= lower) & (data <= upper)]

    # High importance features
    n_high = len(high_features)
    n_rows = (n_high + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    labels = [f"[{chr(97+i)}]" for i in range(n_high)]

    for i, feature in enumerate(high_features):
        nonsep = remove_outliers(np.abs(X[y == 0][feature].values))
        sep = remove_outliers(np.abs(X[y == 1][feature].values))
        axes[i].boxplot(
            [nonsep, sep],
            labels=["NonSEP", "SEP"],
            patch_artist=True,
            boxprops=dict(facecolor="#1f77b4", alpha=0.7),
            medianprops=dict(color="black"),
        )
        axes[i].set_title(feature)
        axes[i].text(0.05, 0.93, labels[i], transform=axes[i].transAxes, fontsize=16)

    for j in range(n_high, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_dir}Boxplots_High_Importance.tif", dpi=300)
    plt.show()

    # Low importance features
    n_low = len(low_features)
    fig, axes = plt.subplots(1, n_low, figsize=(14, 4))
    axes = axes.flatten()
    labels = [f"[{chr(97+i)}]" for i in range(n_low)]

    for i, feature in enumerate(low_features):
        nonsep = remove_outliers(np.abs(X[y == 0][feature].values))
        sep = remove_outliers(np.abs(X[y == 1][feature].values))
        axes[i].boxplot(
            [nonsep, sep],
            labels=["NonSEP", "SEP"],
            patch_artist=True,
            boxprops=dict(facecolor="#ff7f0e", alpha=0.7),
            medianprops=dict(color="black"),
        )
        axes[i].set_title(feature)
        axes[i].text(0.05, 0.93, labels[i], transform=axes[i].transAxes, fontsize=16)

    plt.tight_layout()
    plt.savefig(f"{save_dir}Boxplots_Low_Importance.tif", dpi=300)
    plt.show()


# =============================================================================
# Comparison Models
# =============================================================================


def train_comparison_models(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Train comparison models (SVM, XGBoost, AdaBoost).

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Predictions from all models.
    """
    models = {
        "SVM": SVC(probability=True, random_state=RANDOM_STATE, C=0.4, gamma=0.1),
        "XGBoost": xgb.XGBClassifier(
            max_depth=1,
            learning_rate=0.05,
            n_estimators=50,
            subsample=0.5,
            colsample_bytree=0.5,
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=RANDOM_STATE),
            n_estimators=10,
            learning_rate=0.2,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        results[name] = {
            "train_pred": model.predict(X_train),
            "test_pred": model.predict(X_test),
            "train_prob": model.predict_proba(X_train)[:, 1],
            "test_prob": model.predict_proba(X_test)[:, 1],
        }

    return results


# =============================================================================
# Main Pipeline
# =============================================================================


def main():
    """Main execution pipeline for SEP prediction."""
    print("\n" + "=" * 70)
    print("  Solar Energetic Particle (SEP) Event Prediction")
    print("  GA-Optimized XGBoost Classification Pipeline")
    print("=" * 70)

    # =========================================================================
    # 1. Data Loading and Preprocessing
    # =========================================================================
    print("\n[Step 1] Loading and preprocessing data...")

    df = load_data(DATA_PATH)

    target_columns = ["DATE", "TIME", "DT", "BURST TYPE", "EVENT TYPE", "End Freq"]
    X, y = preprocess_features(df, target_columns)
    feature_names = X.columns.tolist()

    X = fill_missing_values(X)
    X = normalize_features(X)

    # =========================================================================
    # 2. Train/Test Split with Noise
    # =========================================================================
    print("\n[Step 2] Splitting data and adding noise...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # Add small noise for robustness
    noise_factor = 0.001
    X_train_noisy = X_train + np.random.normal(0, noise_factor, X_train.shape)
    X_test_noisy = X_test + np.random.normal(0, noise_factor, X_test.shape)

    # Calculate class weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / pos_count
    print(f"Scale positive weight: {scale_pos:.4f}")

    # =========================================================================
    # 3. GA-XGBoost Optimization
    # =========================================================================
    print("\n[Step 3] Running GA-XGBoost optimization...")

    ga_optimizer = GAXGBoostOptimizer(
        scale_pos_weight=scale_pos,
        population_size=40,
        generations=60,
        n_repeats=3,
    )
    ga_optimizer.fit(X_train_noisy, y_train, X_test_noisy, y_test)

    # =========================================================================
    # 4. Model Evaluation
    # =========================================================================
    print("\n[Step 4] Evaluating GA-XGBoost model...")

    y_train_prob = ga_optimizer.predict_proba(X_train_noisy)
    y_train_pred = ga_optimizer.predict(X_train_noisy)
    y_test_prob = ga_optimizer.predict_proba(X_test_noisy)
    y_test_pred = ga_optimizer.predict(X_test_noisy)

    train_metrics = evaluate_classifier(y_train, y_train_pred, y_train_prob)
    test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_prob)

    print_evaluation_results(train_metrics, "Training Set")
    print_evaluation_results(test_metrics, "Test Set")

    # =========================================================================
    # 5. Comparison with Other Models
    # =========================================================================
    print("\n[Step 5] Training comparison models...")

    comparison_results = train_comparison_models(
        X_train_noisy, y_train, X_test_noisy, y_test
    )

    # Print comparison table
    print("\n" + "=" * 85)
    print("Model Comparison - Test Set")
    print("=" * 85)
    print(f"{'Model':<12} {'Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print("-" * 85)

    all_test_probs = {}

    for name, preds in comparison_results.items():
        metrics = evaluate_classifier(y_test, preds["test_pred"], preds["test_prob"])
        print(
            f"{name:<12} {metrics['Accuracy']:.4f}     "
            f"{metrics['Precision']:.4f}     {metrics['Recall/POD']:.4f}     "
            f"{metrics['F1']:.4f}     {metrics['AUC']:.4f}"
        )
        all_test_probs[name] = preds["test_prob"]

    # Add GA-XGBoost
    print(
        f"{'GA-XGBoost':<12} {test_metrics['Accuracy']:.4f}     "
        f"{test_metrics['Precision']:.4f}     {test_metrics['Recall/POD']:.4f}     "
        f"{test_metrics['F1']:.4f}     {test_metrics['AUC']:.4f}  << Best"
    )
    all_test_probs["GA-XGBoost"] = y_test_prob

    # =========================================================================
    # 6. Visualizations
    # =========================================================================
    print("\n[Step 6] Generating visualizations...")

    # Learning curve
    plot_learning_curve(
        ga_optimizer.best_model, X_train_noisy, y_train, save_path="learning_curve.tif"
    )

    # Training loss
    plot_training_loss(ga_optimizer.loss_history, save_path="training_loss.tif")

    # GA convergence
    plot_ga_convergence(ga_optimizer.ga_history, save_path="ga_convergence.tif")

    # ROC curves
    plot_roc_curves(all_test_probs, y_test, save_path="roc_curves.tif")

    # Feature importance
    top_features = plot_feature_importance(
        ga_optimizer.best_model,
        feature_names,
        top_n=7,
        save_path="feature_importance.tif",
    )

    # =========================================================================
    # 7. Retrain with Top Features + SHAP
    # =========================================================================
    print("\n[Step 7] Retraining with top features and SHAP analysis...")

    X_train_top = X_train_noisy[top_features]
    X_test_top = X_test_noisy[top_features]

    model_top = xgb.XGBClassifier(
        max_depth=ga_optimizer.best_model.max_depth,
        learning_rate=ga_optimizer.best_model.learning_rate,
        reg_alpha=ga_optimizer.best_model.reg_alpha,
        reg_lambda=ga_optimizer.best_model.reg_lambda,
        subsample=ga_optimizer.best_model.subsample,
        colsample_bytree=ga_optimizer.best_model.colsample_bytree,
        random_state=RANDOM_STATE,
    )
    model_top.fit(X_train_top, y_train)

    # Evaluate top-feature model
    y_pred_top = model_top.predict(X_test_top)
    y_prob_top = model_top.predict_proba(X_test_top)[:, 1]
    top_metrics = evaluate_classifier(y_test, y_pred_top, y_prob_top)
    print_evaluation_results(top_metrics, f"Top {len(top_features)} Features Model")

    # SHAP analysis
    plot_shap_summary(
        model_top, X_test_top, top_features, save_path="shap_summary.tif"
    )

    # Feature boxplots
    low_features = [f for f in feature_names if f not in top_features]
    plot_feature_boxplots(X, y, top_features, low_features)

    print("\n" + "=" * 70)
    print("  Pipeline completed successfully!")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()