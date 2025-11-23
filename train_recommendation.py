"""
Training Pipeline for Recommendation System
Train và evaluate các models: Content-Based, GNN, Hybrid
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Add recommendation_system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from recommendation_system.data.preprocessing import DataPreprocessor
from recommendation_system.models.content_based import ContentBasedRecommender
from recommendation_system.models.gnn_model import GNNRecommender
from recommendation_system.models.hybrid_model import HybridRecommender
from recommendation_system.evaluation.metrics import RecommendationEvaluator


def get_or_create_preprocessor():
    """Load preprocessor nếu đã có, nếu không thì tạo mới"""
    base_dir = Path(__file__).parent
    preprocessor_path = base_dir / "recommendation_system" / "data" / "preprocessor.pkl"
    
    if preprocessor_path.exists():
        print(f"📂 Loading existing preprocessor from {preprocessor_path}")
        with open(preprocessor_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("📂 Creating new preprocessor...")
        # Paths
        users_path = base_dir / "exports" / "users.csv"
        products_path = base_dir / "exports" / "products.csv"
        interactions_path = base_dir / "exports" / "interactions.csv"
        
        # Check if files exist
        if not all([users_path.exists(), products_path.exists(), interactions_path.exists()]):
            raise FileNotFoundError(
                f"Missing data files. Please ensure these files exist:\n"
                f"  - {users_path}\n"
                f"  - {products_path}\n"
                f"  - {interactions_path}"
            )
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(
            users_path=str(users_path),
            products_path=str(products_path),
            interactions_path=str(interactions_path)
        )
        
        # Run preprocessing
        preprocessor.preprocess_all()
        
        # Save preprocessor
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"✅ Saved preprocessor to {preprocessor_path}")
        
        return preprocessor


def train_content_based(evaluate=True):
    """Train Content-Based model"""
    print("="*80)
    print("TRAINING CONTENT-BASED MODEL")
    print("="*80)
    
    # Get preprocessor
    preprocessor = get_or_create_preprocessor()
    
    # Train model
    cb_model = ContentBasedRecommender(preprocessor.products_df)
    cb_model.train()
    
    # Save model
    base_dir = Path(__file__).parent
    cb_model_path = base_dir / "recommendation_system" / "models" / "content_based_model.pkl"
    cb_model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cb_model_path, 'wb') as f:
        pickle.dump(cb_model, f)
    print(f"✅ Saved Content-Based model to {cb_model_path}")
    
    # Evaluate if requested
    if evaluate:
        print("\n" + "-"*80)
        print("EVALUATING CONTENT-BASED MODEL")
        print("-"*80)
        evaluator = RecommendationEvaluator(
            test_interactions=preprocessor.test_interactions,
            products_df=preprocessor.products_df,
            train_interactions=preprocessor.train_interactions
        )
        results = evaluator.evaluate_model(
            model=cb_model,
            model_name="Content-Based Filtering",
            users_df=preprocessor.users_df,
            k_values=[10, 20]
        )
        save_evaluation_result(results, base_dir)
    
    return cb_model


def train_gnn(evaluate=True):
    """Train GNN model"""
    print("="*80)
    print("TRAINING GNN MODEL")
    print("="*80)
    
    # Get preprocessor
    preprocessor = get_or_create_preprocessor()
    
    # Train model
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
    gnn_model.train(n_epochs=30, learning_rate=0.001, batch_size=2048)  # Tối ưu: giảm epochs, tăng batch_size
    
    # Save model
    base_dir = Path(__file__).parent
    gnn_model_path = base_dir / "recommendation_system" / "models" / "gnn_model.pkl"
    with open(gnn_model_path, 'wb') as f:
        pickle.dump(gnn_model, f)
    print(f"✅ Saved GNN model to {gnn_model_path}")
    
    # Evaluate if requested
    if evaluate:
        print("\n" + "-"*80)
        print("EVALUATING GNN MODEL")
        print("-"*80)
        evaluator = RecommendationEvaluator(
            test_interactions=preprocessor.test_interactions,
            products_df=preprocessor.products_df,
            train_interactions=preprocessor.train_interactions
        )
        results = evaluator.evaluate_model(
            model=gnn_model,
            model_name="GNN (GraphSAGE)",
            users_df=preprocessor.users_df,
            k_values=[10, 20]
        )
        save_evaluation_result(results, base_dir)
    
    return gnn_model


def train_hybrid(evaluate=True):
    """Train Hybrid model (requires GNN and Content-Based models)"""
    print("="*80)
    print("TRAINING HYBRID MODEL")
    print("="*80)
    
    # Get preprocessor
    preprocessor = get_or_create_preprocessor()
    
    # Load or train required models
    base_dir = Path(__file__).parent
    
    # Load Content-Based model
    cb_model_path = base_dir / "recommendation_system" / "models" / "content_based_model.pkl"
    if cb_model_path.exists():
        print(f"📂 Loading Content-Based model from {cb_model_path}")
        with open(cb_model_path, 'rb') as f:
            cb_model = pickle.load(f)
    else:
        print("⚠️  Content-Based model not found. Training it first...")
        cb_model = train_content_based(evaluate=False)
    
    # Load GNN model
    gnn_model_path = base_dir / "recommendation_system" / "models" / "gnn_model.pkl"
    if gnn_model_path.exists():
        print(f"📂 Loading GNN model from {gnn_model_path}")
        with open(gnn_model_path, 'rb') as f:
            gnn_model = pickle.load(f)
    else:
        print("⚠️  GNN model not found. Training it first...")
        gnn_model = train_gnn(evaluate=False)
    
    # Train hybrid model
    hybrid_model = HybridRecommender(
        gnn_model=gnn_model,
        content_based_model=cb_model,
        alpha=0.5
    )
    hybrid_model.train()
    
    # Save model
    hybrid_model_path = base_dir / "recommendation_system" / "models" / "hybrid_model.pkl"
    with open(hybrid_model_path, 'wb') as f:
        pickle.dump(hybrid_model, f)
    print(f"✅ Saved Hybrid model to {hybrid_model_path}")
    
    # Evaluate if requested
    if evaluate:
        print("\n" + "-"*80)
        print("EVALUATING HYBRID MODEL")
        print("-"*80)
        evaluator = RecommendationEvaluator(
            test_interactions=preprocessor.test_interactions,
            products_df=preprocessor.products_df,
            train_interactions=preprocessor.train_interactions
        )
        results = evaluator.evaluate_model(
            model=hybrid_model,
            model_name="Hybrid (GNN + Content-Based)",
            users_df=preprocessor.users_df,
            k_values=[10, 20]
        )
        save_evaluation_result(results, base_dir)
    
    return hybrid_model


def save_evaluation_result(result, base_dir):
    """Save single evaluation result, merging with existing results if any"""
    results_path = base_dir / "recommendation_system" / "evaluation" / "comparison_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if any
    if results_path.exists():
        existing_df = pd.read_csv(results_path)
        # Remove row with same model_name if exists
        existing_df = existing_df[existing_df['model_name'] != result['model_name']]
        # Append new result
        results_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
    else:
        results_df = pd.DataFrame([result])
    
    # Save
    results_df.to_csv(results_path, index=False)
    print(f"✅ Saved evaluation results to {results_path}")


def train_and_evaluate():
    """Train và evaluate tất cả models"""
    
    print("="*80)
    print("RECOMMENDATION SYSTEM TRAINING PIPELINE")
    print("="*80)
    
    """Train và evaluate tất cả models (legacy function, use individual train functions instead)"""
    print("="*80)
    print("RECOMMENDATION SYSTEM TRAINING PIPELINE (ALL MODELS)")
    print("="*80)
    
    # Get preprocessor
    preprocessor = get_or_create_preprocessor()
    
    # Train all models
    cb_model = train_content_based(evaluate=False)
    gnn_model = train_gnn(evaluate=False)
    hybrid_model = train_hybrid(evaluate=False)
    
    # Evaluate all
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
        result = evaluator.evaluate_model(
            model=model,
            model_name=name,
            users_df=preprocessor.users_df,
            k_values=[10, 20]
        )
        results.append(result)
        save_evaluation_result(result, base_dir)
    
    # Print summary
    results_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TRAINING & EVALUATION COMPLETE")
    print("="*80)
    print("\n📊 Results Summary:")
    print(results_df.to_string(index=False))
    
    return preprocessor, cb_model, gnn_model, hybrid_model, results_df


def main():
    """Main entry point"""
    try:
        train_and_evaluate()
        print("\n✅ Pipeline completed successfully!")
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

