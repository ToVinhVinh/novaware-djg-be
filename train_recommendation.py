import io
import os
import re
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_system.data.preprocessing import DataPreprocessor
from recommendation_system.models.content_based import ContentBasedRecommender
from recommendation_system.models.gnn_model import GNNRecommender
from recommendation_system.models.hybrid_model import HybridRecommender
from recommendation_system.evaluation.metrics import RecommendationEvaluator

def _slugify_model_name(model_name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')

def save_evaluation_log(model_name: str, log_text: str, base_dir: Path):
    if not log_text:
        return
    logs_dir = base_dir / "recommendation_system" / "evaluation" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify_model_name(model_name)
    log_path = logs_dir / f"{slug}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"=== {model_name} Evaluation Log ({timestamp}) ===\n"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(log_text)

def evaluate_and_record(
    evaluator: RecommendationEvaluator,
    model,
    model_name: str,
    users_df: pd.DataFrame,
    base_dir: Path,
    k_values=None
):
    if k_values is None:
        k_values = [10, 20]

    log_buffer = io.StringIO()
    with redirect_stdout(log_buffer):
        results = evaluator.evaluate_model(
            model=model,
            model_name=model_name,
            users_df=users_df,
            k_values=k_values
        )
    save_evaluation_result(results, base_dir)
    save_evaluation_log(model_name, log_buffer.getvalue(), base_dir)
    return results

def delete_old_pkl_files():
    base_dir = Path(__file__).parent
    files_to_delete = [
        "recommendation_system/data/preprocessor.pkl",
        "recommendation_system/models/content_based_model.pkl",
        "recommendation_system/models/gnn_model.pkl",
        "recommendation_system/models/hybrid_model.pkl",
        "artifacts/cbf/artifacts.pkl",
        "artifacts/gnn/artifacts.pkl",
        "artifacts/hybrid/artifacts.pkl",
    ]

    deleted_count = 0
    for file_path in files_to_delete:
        full_path = base_dir / file_path
        if full_path.exists():
            try:
                full_path.unlink()
                print(f"[DELETE] Deleted: {file_path}")
                deleted_count += 1
            except Exception as e:
                print(f"[WARN] Error deleting {file_path}: {e}")

    if deleted_count > 0:
        print(f"[OK] Deleted {deleted_count} old pkl file(s)\n")

def get_or_create_preprocessor(force_retrain=False):

    base_dir = Path(__file__).parent
    preprocessor_path = base_dir / "recommendation_system" / "data" / "preprocessor.pkl"

    if force_retrain and preprocessor_path.exists():
        print(f"[FORCE RETRAIN] Deleting existing preprocessor...")
        preprocessor_path.unlink()

    if preprocessor_path.exists():
        print(f"[LOAD] Loading existing preprocessor from {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("[LOAD] Creating new preprocessor...")
        users_path = base_dir / "exports" / "users.csv"
        products_path = base_dir / "exports" / "products.csv"
        interactions_path = base_dir / "exports" / "interactions.csv"

        if not all([users_path.exists(), products_path.exists(), interactions_path.exists()]):
            raise FileNotFoundError(
                f"Missing data files. Please ensure these files exist:\n"
                f"  - {users_path}\n"
                f"  - {products_path}\n"
                f"  - {interactions_path}"
            )

        preprocessor = DataPreprocessor(
            users_path=str(users_path),
            products_path=str(products_path),
            interactions_path=str(interactions_path)
        )

        preprocessor.preprocess_all()

        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"[OK] Saved preprocessor to {preprocessor_path}")

        return preprocessor

def train_content_based(evaluate=True, force_retrain=False):

    print("="*80)
    print("TRAINING CONTENT-BASED MODEL")
    print("="*80)

    if force_retrain:
        base_dir = Path(__file__).parent
        cb_model_path = base_dir / "recommendation_system" / "models" / "content_based_model.pkl"
        if cb_model_path.exists():
            cb_model_path.unlink()
            print("[FORCE RETRAIN] Deleted old Content-Based model")

    preprocessor = get_or_create_preprocessor(force_retrain=force_retrain)

    cb_model = ContentBasedRecommender(preprocessor.products_df)
    cb_model.train()

    base_dir = Path(__file__).parent
    cb_model_path = base_dir / "recommendation_system" / "models" / "content_based_model.pkl"
    cb_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cb_model_path, 'wb') as f:
        pickle.dump(cb_model, f)
    print(f"[OK] Saved Content-Based model to {cb_model_path}")

    if evaluate:
        print("\n" + "-"*80)
        print("EVALUATING CONTENT-BASED MODEL")
        print("-"*80)
        evaluator = RecommendationEvaluator(
            test_interactions=preprocessor.test_interactions,
            products_df=preprocessor.products_df,
            train_interactions=preprocessor.train_interactions
        )
        evaluate_and_record(
            evaluator=evaluator,
            model=cb_model,
            model_name="Content-Based Filtering",
            users_df=preprocessor.users_df,
            base_dir=base_dir,
            k_values=[10, 20]
        )

    return cb_model

def train_gnn(evaluate=True, force_retrain=False):

    print("="*80)
    print("TRAINING GNN MODEL")
    print("="*80)

    if force_retrain:
        base_dir = Path(__file__).parent
        gnn_model_path = base_dir / "recommendation_system" / "models" / "gnn_model.pkl"
        if gnn_model_path.exists():
            gnn_model_path.unlink()
            print("[FORCE RETRAIN] Deleted old GNN model")

    preprocessor = get_or_create_preprocessor(force_retrain=force_retrain)

    gnn_model = GNNRecommender(
        users_df=preprocessor.users_df,
        products_df=preprocessor.products_df,
        train_interactions=preprocessor.train_interactions,
        embedding_dim=64,
        hidden_dim=128,
        n_layers=2,
        dropout=0.3,
        device='cpu'
    )
    gnn_model.train(n_epochs=30, learning_rate=0.001, batch_size=2048)

    base_dir = Path(__file__).parent
    gnn_model_path = base_dir / "recommendation_system" / "models" / "gnn_model.pkl"
    with open(gnn_model_path, 'wb') as f:
        pickle.dump(gnn_model, f)
    print(f"[OK] Saved GNN model to {gnn_model_path}")

    if evaluate:
        print("\n" + "-"*80)
        print("EVALUATING GNN MODEL")
        print("-"*80)
        evaluator = RecommendationEvaluator(
            test_interactions=preprocessor.test_interactions,
            products_df=preprocessor.products_df,
            train_interactions=preprocessor.train_interactions
        )
        evaluate_and_record(
            evaluator=evaluator,
            model=gnn_model,
            model_name="GNN (GraphSAGE)",
            users_df=preprocessor.users_df,
            base_dir=base_dir,
            k_values=[10, 20]
        )

    return gnn_model

def train_hybrid(evaluate=True, force_retrain=False):

    print("="*80)
    print("TRAINING HYBRID MODEL")
    print("="*80)

    if force_retrain:
        base_dir = Path(__file__).parent
        hybrid_model_path = base_dir / "recommendation_system" / "models" / "hybrid_model.pkl"
        if hybrid_model_path.exists():
            hybrid_model_path.unlink()
            print("[FORCE RETRAIN] Deleted old Hybrid model")

    preprocessor = get_or_create_preprocessor(force_retrain=force_retrain)

    base_dir = Path(__file__).parent

    cb_model_path = base_dir / "recommendation_system" / "models" / "content_based_model.pkl"
    if cb_model_path.exists() and not force_retrain:
        print(f"[LOAD] Loading Content-Based model from {cb_model_path}")
        with open(cb_model_path, 'rb') as f:
            cb_model = pickle.load(f)
    else:
        print("[WARN] Content-Based model not found or force retrain. Training it first...")
        cb_model = train_content_based(evaluate=False, force_retrain=force_retrain)

    gnn_model_path = base_dir / "recommendation_system" / "models" / "gnn_model.pkl"
    if gnn_model_path.exists() and not force_retrain:
        print(f"[LOAD] Loading GNN model from {gnn_model_path}")
        with open(gnn_model_path, 'rb') as f:
            gnn_model = pickle.load(f)
    else:
        print("[WARN] GNN model not found or force retrain. Training it first...")
        gnn_model = train_gnn(evaluate=False, force_retrain=force_retrain)

    hybrid_model = HybridRecommender(
        gnn_model=gnn_model,
        content_based_model=cb_model,
        alpha=0.5
    )
    hybrid_model.train()

    hybrid_model_path = base_dir / "recommendation_system" / "models" / "hybrid_model.pkl"
    with open(hybrid_model_path, 'wb') as f:
        pickle.dump(hybrid_model, f)
    print(f"[OK] Saved Hybrid model to {hybrid_model_path}")

    if evaluate:
        print("\n" + "-"*80)
        print("EVALUATING HYBRID MODEL")
        print("-"*80)
        evaluator = RecommendationEvaluator(
            test_interactions=preprocessor.test_interactions,
            products_df=preprocessor.products_df,
            train_interactions=preprocessor.train_interactions
        )
        evaluate_and_record(
            evaluator=evaluator,
            model=hybrid_model,
            model_name="Hybrid (GNN + Content-Based)",
            users_df=preprocessor.users_df,
            base_dir=base_dir,
            k_values=[10, 20]
        )

    return hybrid_model

def save_evaluation_result(result, base_dir):
    results_path = base_dir / "recommendation_system" / "evaluation" / "comparison_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        existing_df = existing_df[existing_df['model_name'] != result['model_name']]
        results_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
    else:
        results_df = pd.DataFrame([result])

    results_df.to_csv(results_path, index=False)
    print(f"[OK] Saved evaluation results to {results_path}")

def train_and_evaluate(force_retrain=False):

    print("="*80)
    print("RECOMMENDATION SYSTEM TRAINING PIPELINE")
    print("="*80)

    if force_retrain:
        print("[FORCE RETRAIN] FORCE RETRAIN MODE: Deleting all old models and preprocessor")
        print("="*80)
        delete_old_pkl_files()

    preprocessor = get_or_create_preprocessor(force_retrain=force_retrain)

    cb_model = train_content_based(evaluate=False, force_retrain=force_retrain)
    gnn_model = train_gnn(evaluate=False, force_retrain=force_retrain)
    hybrid_model = train_hybrid(evaluate=False, force_retrain=force_retrain)

    base_dir = Path(__file__).parent
    evaluator = RecommendationEvaluator(
        test_interactions=preprocessor.test_interactions,
        products_df=preprocessor.products_df,
        train_interactions=preprocessor.train_interactions
    )

    results = []
    for model, name in [(cb_model, "Content-Based Filtering"),
                        (gnn_model, "GNN (GraphSAGE)"),
                        (hybrid_model, "Hybrid (GNN + Content-Based)")]:
        result = evaluate_and_record(
            evaluator=evaluator,
            model=model,
            model_name=name,
            users_df=preprocessor.users_df,
            base_dir=base_dir,
            k_values=[10, 20]
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TRAINING & EVALUATION COMPLETE")
    print("="*80)
    print("\n[INFO] Results Summary:")
    print(results_df.to_string(index=False))

    return preprocessor, cb_model, gnn_model, hybrid_model, results_df

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train recommendation system models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_recommendation.py

  python train_recommendation.py --force-retrain
        """
    )

    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retrain all models (delete existing models first)'
    )

    args = parser.parse_args()

    train_and_evaluate(force_retrain=args.force_retrain)

if __name__ == "__main__":
    main()
