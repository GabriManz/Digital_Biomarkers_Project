import os
import re
import subprocess
import json
import pandas as pd
import numpy as np

# Paths
ROOT_DIR = r"c:\DATA\01_Proyectos\Master\Digital_Biomarkers\Project"
STEP5_PATH = os.path.join(ROOT_DIR, "src", "step5_features.py")
CACHE_DIR = os.path.join(ROOT_DIR, "outputs", "results", "step5")
CLASSIF_DIR = os.path.join(ROOT_DIR, "outputs", "results", "step6_loso")
DELTA_CAS_DIR = os.path.join(ROOT_DIR, "outputs", "results", "step7b")
PYTHON_EXE = os.path.join(ROOT_DIR, ".venv", "Scripts", "python.exe")

def set_preprocessing_flags(bandpass: bool, preemphasis: bool):
    print(f"\nSetting APPLY_BANDPASS_SEGMENT = {bandpass}, APPLY_PREEMPHASIS = {preemphasis} in {STEP5_PATH}")
    with open(STEP5_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace APPLY_BANDPASS_SEGMENT
    content = re.sub(
        r"APPLY_BANDPASS_SEGMENT\s*=\s*(True|False)",
        f"APPLY_BANDPASS_SEGMENT = {bandpass}",
        content
    )
    # Replace APPLY_PREEMPHASIS
    content = re.sub(
        r"APPLY_PREEMPHASIS\s*=\s*(True|False)",
        f"APPLY_PREEMPHASIS     = {preemphasis}",
        content
    )
    
    with open(STEP5_PATH, "w", encoding="utf-8") as f:
        f.write(content)

def clear_feature_cache():
    print("Clearing step5 feature cache files...")
    if os.path.exists(CACHE_DIR):
        for fname in os.listdir(CACHE_DIR):
            if fname.endswith(".npy") or fname.endswith(".json"):
                path = os.path.join(CACHE_DIR, fname)
                try:
                    os.remove(path)
                    print(f"  Deleted: {fname}")
                except Exception as e:
                    print(f"  Error deleting {fname}: {e}")

def run_command(args, label):
    print(f"\nRunning {label}...")
    res = subprocess.run(args, cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"Error running {label}!")
        print("STDOUT:")
        print(res.stdout[-1000:])
        print("STDERR:")
        print(res.stderr[-1000:])
        raise RuntimeError(f"Command failed with code {res.returncode}")
    print(f"{label} completed successfully.")
    return res.stdout

def extract_bdr_results():
    # Load step6 results for Ensemble
    # We want to find: segment AUC, patient Accuracy, patient AUC
    # We will read outputs/results/step6_loso/ensemble_loso_results.csv or others
    # and print out the summary table.
    
    models = ["SVM", "RF", "XGB", "Ensemble"]
    summary = {}
    
    for m in models:
        # Segment-level metrics
        seg_file = os.path.join(CLASSIF_DIR, f"{m.lower()}_loso_results.csv")
        pat_file = os.path.join(CLASSIF_DIR, f"{m.lower()}_loso_patient_results.csv")
        
        if not os.path.exists(seg_file) or not os.path.exists(pat_file):
            print(f"Warning: missing results for model {m}")
            continue
            
        df_seg = pd.read_csv(seg_file)
        df_pat = pd.read_csv(pat_file)
        
        mean_seg_auc = df_seg["auc"].mean()
        std_seg_auc = df_seg["auc"].std()
        
        # Patient-level Acc
        pat_acc = df_pat["correct"].mean()
        
        # Patient-level AUC (using patient probabilities)
        # We need actual labels and probabilities
        pat_labels = df_pat["label_patient"].values
        pat_probs = df_pat["prob_patient"].values
        
        from sklearn.metrics import roc_auc_score
        try:
            pat_auc = roc_auc_score(pat_labels, pat_probs)
        except ValueError:
            pat_auc = 0.0
            
        # Get Delta_CAS LogReg and SVM-Lin patient-level results from step7b results file
        # step7b_delta_cas.py saves results to outputs/results/step7b/patient_delta_cas_{model}.csv
        # but the summary metrics are printed to stdout, which we can parse or compute from patient_delta_cas_{model}.csv
        # Let's load patient_delta_cas_{model}.csv to calculate them ourselves!
        # patient_delta_cas_{model}.csv columns: subject_id, bdr_label, type, delta_cas, cas_rate_pre, cas_rate_post, prob_mean_pre, prob_mean_post, prob_delta, iqr_prob_pre, iqr_prob_post
        
        pat_df_path = os.path.join(DELTA_CAS_DIR, f"patient_delta_cas_{m}.csv")
        lr_acc, lr_auc, svm_acc, svm_auc = 0.0, 0.0, 0.0, 0.0
        
        if os.path.exists(pat_df_path):
            df_pat_bdr = pd.read_csv(pat_df_path)
            # Filter to patients only
            df_pat_bdr = df_pat_bdr[df_pat_bdr["type"] == "patient"].reset_index(drop=True)
            
            # Let's run LOSO on this patient dataset to compute LR and SVM-Lin BDR classification metrics
            # (exactly as step7b_delta_cas.py does)
            feature_cols = [
                "delta_cas", "cas_rate_pre", "cas_rate_post",
                "prob_mean_pre", "prob_mean_post", "prob_delta",
                "iqr_prob_pre", "iqr_prob_post"
            ]
            
            X = df_pat_bdr[feature_cols].values.astype(float)
            y = (df_pat_bdr["bdr_label"] == "BDR+").astype(int).values
            groups = df_pat_bdr["subject_num"].values
            
            from sklearn.model_selection import LeaveOneGroupOut
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            from sklearn.svm import SVC
            
            loso = LeaveOneGroupOut()
            
            # LogReg
            lr_corrects = []
            lr_probs = []
            
            # SVM
            svm_corrects = []
            svm_probs = []
            
            for train_idx, test_idx in loso.split(X, y, groups):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]
                
                sc = StandardScaler()
                X_tr_sc = sc.fit_transform(X_tr)
                X_te_sc = sc.transform(X_te)
                
                # LR
                lr = LogisticRegression(C=0.1, class_weight="balanced", random_state=42)
                lr.fit(X_tr_sc, y_tr)
                lr_pred = lr.predict(X_te_sc)[0]
                lr_prob = lr.predict_proba(X_te_sc)[0, 1]
                lr_corrects.append(int(lr_pred == y_te[0]))
                lr_probs.append(lr_prob)
                
                # SVM
                svm = SVC(kernel="linear", C=0.1, class_weight="balanced", probability=True, random_state=42)
                svm.fit(X_tr_sc, y_tr)
                svm_pred = svm.predict(X_te_sc)[0]
                svm_prob = svm.predict_proba(X_te_sc)[0, 1]
                svm_corrects.append(int(svm_pred == y_te[0]))
                svm_probs.append(svm_prob)
                
            lr_acc = np.mean(lr_corrects)
            try:
                lr_auc = roc_auc_score(y, lr_probs)
            except ValueError:
                lr_auc = 0.0
                
            svm_acc = np.mean(svm_corrects)
            try:
                svm_auc = roc_auc_score(y, svm_probs)
            except ValueError:
                svm_auc = 0.0
                
        summary[m] = {
            "seg_auc": f"{mean_seg_auc:.3f} ± {std_seg_auc:.3f}",
            "pat_acc": f"{pat_acc:.3f}",
            "pat_auc": f"{pat_auc:.3f}",
            "bdr_lr_acc": f"{lr_acc:.3f}",
            "bdr_lr_auc": f"{lr_auc:.3f}",
            "bdr_svm_acc": f"{svm_acc:.3f}",
            "bdr_svm_auc": f"{svm_auc:.3f}"
        }
        
    return summary

def main():
    experiments = [
        {"name": "Baseline (No preprocessing)", "bandpass": False, "preemphasis": False},
        {"name": "Pre-emphasis only", "bandpass": False, "preemphasis": True},
        {"name": "Pre-emphasis + Bandpass", "bandpass": True, "preemphasis": True}
    ]
    
    results = {}
    
    for exp in experiments:
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT: {exp['name']}")
        print("="*80)
        
        # Configure flags
        set_preprocessing_flags(exp["bandpass"], exp["preemphasis"])
        
        # Clear cache
        clear_feature_cache()
        
        # Run step5
        run_command([PYTHON_EXE, "src/step5_features.py"], f"Feature Extraction ({exp['name']})")
        
        # Run step6
        run_command([PYTHON_EXE, "src/step6_classification_loso.py"], f"Classification LOSO ({exp['name']})")
        
        # Run step7b
        run_command([PYTHON_EXE, "src/step7b_delta_cas.py"], f"Delta_CAS BDR ({exp['name']})")
        
        # Collect results
        results[exp["name"]] = extract_bdr_results()
        
    # Write summary report
    print("\n" + "="*100)
    print("ALL EXPERIMENTS COMPLETED. SUMMARY OF RESULTS:")
    print("="*100)
    
    # Save results to JSON for parsing later
    with open(os.path.join(ROOT_DIR, "outputs", "results", "experiments_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # Print Markdown table
    for model in ["SVM", "RF", "XGB", "Ensemble"]:
        print(f"\n### Model: {model}")
        print("| Experiment | Seg AUC | Pat Acc | Pat AUC | BDR LR Acc | BDR LR AUC | BDR SVM Acc | BDR SVM AUC |")
        print("|------------|---------|---------|---------|------------|------------|-------------|-------------|")
        for exp_name in results:
            r = results[exp_name][model]
            print(f"| {exp_name:<26} | {r['seg_auc']} | {r['pat_acc']} | {r['pat_auc']} | {r['bdr_lr_acc']} | {r['bdr_lr_auc']} | {r['bdr_svm_acc']} | {r['bdr_svm_auc']} |")

if __name__ == "__main__":
    main()
