
# ============================================================
# Khetha Milestone 3
# Ensemble Optimisation & Nested Hyperparameter Tuning
# File: khetha_ensemble_optimizer.py
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from scipy.stats import loguniform, randint, t

from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold,
    train_test_split
)
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import check_is_fitted

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    TPESampler = None
    OPTUNA_AVAILABLE = False


class KhethaEnsembleOptimizer:
    """
    Manages ensemble training, hyperparameter optimisation,
    bias-variance diagnosis and nested cross-validation.
    """

    def __init__(self, X_train, y_train, X_test, y_test):

        self.X_train = X_train
        self.y_train = np.asarray(y_train).ravel()

        self.X_test = X_test
        self.y_test = np.asarray(y_test).ravel()

        self.tuned_model = None
        self.ensembles = {}
        self.search_results = {}
        self.convergence_trajectory = []
        self.weight_trajectories = {}
        self.early_stopping_results = {}
        self.nested_cv_results = {}

        self.random_state = 42

    # --------------------------------------------------------
    # 1. Bias-Variance Diagnosis
    # --------------------------------------------------------

    def diagnose_bias_variance(
        self,
        tree_score=None,
        train_score=None,
        validation_score=None,
        noise_level="medium"
    ):
        """
        Determines whether Bagging or Boosting is more suitable.

        High variance -> Random Forest (Bagging)
        High bias -> Gradient Boosting (Boosting)
        High noise -> Prefer a more robust Bagging approach.
        """

        if train_score is None or validation_score is None:

            if tree_score is not None:
                train_score = tree_score
                validation_score = tree_score
            else:
                tree = DecisionTreeClassifier(
                    random_state=self.random_state
                )

                scores = cross_val_score(
                    tree,
                    self.X_train,
                    self.y_train,
                    cv=3,
                    scoring="accuracy"
                )

                validation_score = np.mean(scores)

                tree.fit(self.X_train, self.y_train)
                train_score = tree.score(
                    self.X_train,
                    self.y_train
                )

        bias_gap = max(0, 1 - train_score)
        variance_gap = max(0, train_score - validation_score)

        if noise_level == "high":
            recommendation = "random_forest"

        elif variance_gap > bias_gap:
            recommendation = "random_forest"

        else:
            recommendation = "gradient_boosting"

        return {
            "bias_gap": float(bias_gap),
            "variance_gap": float(variance_gap),
            "noise_level": noise_level,
            "recommended_strategy": recommendation
        }

    # --------------------------------------------------------
    # 2. Ensemble Construction
    # --------------------------------------------------------

    def build_ensembles(self, strategy="random_forest"):
        """
        Configures Random Forest, Gradient Boosting and AdaBoost.

        Random Forest:
            Uses OOB scoring to estimate generalisation.

        Gradient Boosting:
            Uses staged predictions for early stopping.

        AdaBoost:
            Uses decision stumps and tracks estimator weights.
        """

        # Single decision tree baseline
        self.ensembles["decision_tree"] = DecisionTreeClassifier(
            max_depth=None,
            random_state=self.random_state
        )

        # Random Forest with OOB scoring
        self.ensembles["random_forest"] = RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            oob_score=True,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state
        )

        # Gradient Boosting
        self.ensembles["gradient_boosting"] = (
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                subsample=0.8,
                max_depth=3,
                random_state=self.random_state
            )
        )

        # AdaBoost with decision stump base learners
        stump = DecisionTreeClassifier(
            max_depth=1,
            random_state=self.random_state
        )

        try:
            self.ensembles["adaboost"] = AdaBoostClassifier(
                estimator=stump,
                n_estimators=100,
                learning_rate=0.5,
                random_state=self.random_state
            )
        except TypeError:
            # Compatibility with older scikit-learn versions
            self.ensembles["adaboost"] = AdaBoostClassifier(
                base_estimator=stump,
                n_estimators=100,
                learning_rate=0.5,
                random_state=self.random_state
            )

        # Automatically diagnose the best strategy if requested
        if strategy == "auto":
            diagnosis = self.diagnose_bias_variance()
            strategy = diagnosis["recommended_strategy"]
            self.bias_variance_diagnosis = diagnosis

        if strategy not in self.ensembles:
            raise ValueError(
                "Strategy must be random_forest, "
                "gradient_boosting, adaboost, decision_tree or auto."
            )

        # Fit the selected model
        if strategy == "gradient_boosting":
            self.tuned_model = self._fit_gradient_boosting_early_stopping()

        else:
            self.tuned_model = self.ensembles[strategy]

            self.tuned_model.fit(
                self.X_train,
                self.y_train
            )

        # Track AdaBoost estimator weight trajectories
        if strategy == "adaboost":
            weights = self.tuned_model.estimator_weights_

            self.weight_trajectories["adaboost"] = (
                np.cumsum(weights).tolist()
            )

        return self.tuned_model

    # --------------------------------------------------------
    # 3. Gradient Boosting Early Stopping
    # --------------------------------------------------------

    def _fit_gradient_boosting_early_stopping(self):
        """
        Uses staged_predict() to monitor validation accuracy.

        Selects the best number of boosting stages and retrains
        the final model using the complete training dataset.
        """

        X_subtrain, X_validation, y_subtrain, y_validation = (
            train_test_split(
                self.X_train,
                self.y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=self.y_train
            )
        )

        candidate = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            max_depth=3,
            random_state=self.random_state
        )

        candidate.fit(X_subtrain, y_subtrain)

        best_score = -np.inf
        best_iteration = 1
        validation_scores = []

        for iteration, predictions in enumerate(
            candidate.staged_predict(X_validation),
            start=1
        ):

            score = accuracy_score(
                y_validation,
                predictions
            )

            validation_scores.append(float(score))

            if score > best_score:
                best_score = score
                best_iteration = iteration

        # Refit using all available training observations
        final_model = GradientBoostingClassifier(
            n_estimators=best_iteration,
            learning_rate=0.05,
            subsample=0.8,
            max_depth=3,
            random_state=self.random_state
        )

        final_model.fit(
            self.X_train,
            self.y_train
        )

        self.early_stopping_results = {
            "best_iteration": best_iteration,
            "best_validation_score": float(best_score),
            "validation_scores": validation_scores
        }

        return final_model

    # --------------------------------------------------------
    # 4. Model Comparison
    # --------------------------------------------------------

    def compare_ensembles(self):
        """
        Compares a single tree against ensemble models.
        """

        results = {}

        for name, model in self.ensembles.items():

            fitted = clone(model)

            fitted.fit(
                self.X_train,
                self.y_train
            )

            predictions = fitted.predict(self.X_test)

            results[name] = {
                "accuracy": float(
                    accuracy_score(self.y_test, predictions)
                )
            }

            if name == "random_forest":
                results[name]["oob_score"] = float(
                    fitted.oob_score_
                )

        return results

    # --------------------------------------------------------
    # 5. Grid Search
    # --------------------------------------------------------

    def run_grid_search(self, param_grid, cv=5):
        """
        Performs exhaustive hyperparameter optimisation.

        Returns best parameters, best score and CV results.
        """

        if self.tuned_model is None:
            self.build_ensembles("random_forest")

        cv_strategy = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        grid_search = GridSearchCV(
            estimator=clone(self.tuned_model),
            param_grid=param_grid,
            scoring="accuracy",
            cv=cv_strategy,
            n_jobs=-1,
            return_train_score=True,
            refit=True
        )

        grid_search.fit(
            self.X_train,
            self.y_train
        )

        self.tuned_model = grid_search.best_estimator_

        results = {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_),
            "cv_results": grid_search.cv_results_
        }

        self.search_results["grid_search"] = results

        return results

    # --------------------------------------------------------
    # 6. Randomised Search
    # --------------------------------------------------------

    def run_randomized_search(
        self,
        param_distributions=None,
        n_iter=50,
        cv=5
    ):
        """
        Uses scipy statistical distributions to explore
        hyperparameter combinations.

        Tracks the best score after every sampled iteration.
        """

        if self.tuned_model is None:
            self.build_ensembles("random_forest")

        if param_distributions is None:

            param_distributions = {
                "n_estimators": randint(50, 301),
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": randint(2, 15),
                "min_samples_leaf": randint(1, 8),
                "max_features": ["sqrt", "log2"],
            }

            if isinstance(
                self.tuned_model,
                GradientBoostingClassifier
            ):
                param_distributions["learning_rate"] = loguniform(
                    0.01, 0.3
                )

        cv_strategy = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        randomized_search = RandomizedSearchCV(
            estimator=clone(self.tuned_model),
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="accuracy",
            cv=cv_strategy,
            random_state=self.random_state,
            n_jobs=-1,
            return_train_score=True,
            refit=True
        )

        randomized_search.fit(
            self.X_train,
            self.y_train
        )

        # Calculate convergence trajectory
        scores = randomized_search.cv_results_["mean_test_score"]

        trajectory = []
        best_so_far = -np.inf

        for score in scores:

            best_so_far = max(best_so_far, score)

            trajectory.append(float(best_so_far))

        self.convergence_trajectory = trajectory

        self.tuned_model = randomized_search.best_estimator_

        results = {
            "best_params": randomized_search.best_params_,
            "best_score": float(randomized_search.best_score_),
            "cv_results": randomized_search.cv_results_,
            "convergence_trajectory": trajectory
        }

        self.search_results["randomized_search"] = results

        return results

    # --------------------------------------------------------
    # 7. Bayesian Optimisation Using Optuna
    # --------------------------------------------------------

    def run_bayesian_optimization(self, n_trials=40):
        """
        Uses Optuna and the Tree-structured Parzen Estimator
        to optimise Random Forest hyperparameters.
        """

        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required. Install it using: "
                "pip install optuna"
            )

        cv_strategy = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.random_state
        )

        def objective(trial):

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 50, 300
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", 3, 30
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 15
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", 1, 8
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2"]
                )
            }

            model = RandomForestClassifier(
                **params,
                oob_score=True,
                bootstrap=True,
                n_jobs=-1,
                random_state=self.random_state
            )

            scores = cross_val_score(
                model,
                self.X_train,
                self.y_train,
                cv=cv_strategy,
                scoring="accuracy",
                n_jobs=-1
            )

            return float(np.mean(scores))

        # TPE sampler guides the search
        sampler = TPESampler(
            seed=self.random_state
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        study.optimize(
            objective,
            n_trials=n_trials
        )

        # Train final model with best parameters
        best_model = RandomForestClassifier(
            **study.best_params,
            oob_score=True,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.random_state
        )

        best_model.fit(
            self.X_train,
            self.y_train
        )

        self.tuned_model = best_model

        results = {
            "best_params": study.best_params,
            "best_score": float(study.best_value),
            "study": study,
            "trials": len(study.trials),
            "convergence_trajectory": [
                float(trial.value)
                for trial in study.trials
                if trial.value is not None
            ]
        }

        self.search_results["bayesian_optimization"] = results

        return results

    # --------------------------------------------------------
    # 8. Nested Cross-Validation
    # --------------------------------------------------------

    def nested_cross_validation(
        self,
        outer_cv=5,
        inner_cv=3
    ):
        """
        Implements leakage-safe nested cross-validation.

        Outer loop:
            Estimates unbiased generalisation performance.

        Inner loop:
            Performs hyperparameter tuning.

        Reports:
            Mean accuracy and 95% confidence interval.
        """

        X = self.X_train
        y = self.y_train

        outer_strategy = StratifiedKFold(
            n_splits=outer_cv,
            shuffle=True,
            random_state=self.random_state
        )

        inner_strategy = StratifiedKFold(
            n_splits=inner_cv,
            shuffle=True,
            random_state=self.random_state
        )

        # Use a fresh estimator for every outer fold
        scores = []
        fold_results = []

        for fold, (train_indices, validation_indices) in enumerate(
            outer_strategy.split(X, y),
            start=1
        ):

            X_outer_train = (
                X.iloc[train_indices]
                if hasattr(X, "iloc")
                else X[train_indices]
            )

            X_outer_validation = (
                X.iloc[validation_indices]
                if hasattr(X, "iloc")
                else X[validation_indices]
            )

            y_outer_train = y[train_indices]
            y_outer_validation = y[validation_indices]

            # All tuning happens exclusively inside outer training data
            estimator = RandomForestClassifier(
                oob_score=True,
                bootstrap=True,
                n_jobs=-1,
                random_state=self.random_state
            )

            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, None],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            }

            inner_search = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring="accuracy",
                cv=inner_strategy,
                n_jobs=-1,
                refit=True
            )

            inner_search.fit(
                X_outer_train,
                y_outer_train
            )

            # Evaluate only on the untouched outer validation fold
            predictions = inner_search.best_estimator_.predict(
                X_outer_validation
            )

            fold_score = accuracy_score(
                y_outer_validation,
                predictions
            )

            scores.append(float(fold_score))

            fold_results.append({
                "fold": fold,
                "score": float(fold_score),
                "best_params": inner_search.best_params_
            })

        scores = np.asarray(scores)

        mean_score = float(np.mean(scores))

        if len(scores) > 1:

            standard_error = (
                np.std(scores, ddof=1) / np.sqrt(len(scores))
            )

            critical_value = t.ppf(
                0.975,
                df=len(scores) - 1
            )

            margin = critical_value * standard_error

        else:
            margin = 0.0

        lower = max(0.0, mean_score - margin)
        upper = min(1.0, mean_score + margin)

        results = {
            "mean_score": mean_score,
            "ci_95": (float(lower), float(upper)),
            "fold_scores": scores.tolist(),
            "fold_results": fold_results
        }

        self.nested_cv_results = results

        return results

    # --------------------------------------------------------
    # 9. Final Model Evaluation
    # --------------------------------------------------------

    def evaluate_model(self):

        if self.tuned_model is None:
            raise ValueError(
                "No trained model found. Build an ensemble first."
            )

        predictions = self.tuned_model.predict(
            self.X_test
        )

        return {
            "accuracy": float(
                accuracy_score(self.y_test, predictions)
            )
        }


# ============================================================
# 10. Standalone Execution
# ============================================================

def main():

    print("=" * 65)
    print("KHETHA - MILESTONE 3")
    print("ENSEMBLE OPTIMISATION & NESTED TUNING")
    print("=" * 65)

    # Milestone 1 and 2 integration points.
    # The previous modules are optional so this file can also
    # execute independently in a clean environment.

    try:
        import khetha_data_pipeline
        print("[INFO] Milestone 1 module detected.")
    except ImportError:
        print("[INFO] Milestone 1 module not detected.")

    try:
        import khetha_baseline_models
        print("[INFO] Milestone 2 module detected.")
    except ImportError:
        print("[INFO] Milestone 2 module not detected.")

    # Demonstration dataset for standalone execution.
    # Replace this with the processed Milestone 1 dataset
    # when integrating the complete project.

    from sklearn.datasets import make_classification

    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    optimizer = KhethaEnsembleOptimizer(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # Build ensembles
    print("\n[1] Building Random Forest...")

    optimizer.build_ensembles("random_forest")

    print(
        "Random Forest OOB Score:",
        optimizer.tuned_model.oob_score_
    )

    # Bias-variance diagnosis
    print("\n[2] Bias-Variance Diagnosis...")

    diagnosis = optimizer.diagnose_bias_variance()

    print(diagnosis)

    # Ensemble comparison
    print("\n[3] Comparing Ensemble Models...")

    comparison = optimizer.compare_ensembles()

    for name, result in comparison.items():
        print(name, result)

    # Grid Search
    print("\n[4] Running Grid Search...")

    grid_results = optimizer.run_grid_search(
        param_grid={
            "n_estimators": [100, 150],
            "max_depth": [5, 10, None],
            "max_features": ["sqrt", "log2"]
        },
        cv=3
    )

    print("Best Grid Parameters:", grid_results["best_params"])
    print("Best Grid Score:", grid_results["best_score"])

    # Randomised Search
    print("\n[5] Running Randomised Search...")

    random_results = optimizer.run_randomized_search(
        param_distributions={
            "n_estimators": randint(50, 200),
            "max_depth": [5, 10, 15, None],
            "max_features": ["sqrt", "log2"]
        },
        n_iter=10,
        cv=3
    )

    print("Best Random Parameters:", random_results["best_params"])
    print("Best Random Score:", random_results["best_score"])

    # Bayesian Optimisation
    print("\n[6] Running Bayesian Optimisation...")

    if OPTUNA_AVAILABLE:

        bayesian_results = optimizer.run_bayesian_optimization(
            n_trials=10
        )

        print(
            "Best Bayesian Parameters:",
            bayesian_results["best_params"]
        )

        print(
            "Best Bayesian Score:",
            bayesian_results["best_score"]
        )

    else:
        print(
            "[WARNING] Optuna is not installed. "
            "Install using: pip install optuna"
        )

    # Nested Cross-Validation
    print("\n[7] Running Nested Cross-Validation...")

    nested_results = optimizer.nested_cross_validation(
        outer_cv=5,
        inner_cv=3
    )

    print("Mean Score:", nested_results["mean_score"])

    print(
        "95% Confidence Interval:",
        nested_results["ci_95"]
    )

    # Final evaluation
    print("\n[8] Final Model Evaluation...")

    final_results = optimizer.evaluate_model()

    print("Test Accuracy:", final_results["accuracy"])

    print("\n" + "=" * 65)
    print("MILESTONE 3 EXECUTION COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
