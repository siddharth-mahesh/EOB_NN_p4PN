import optuna
import pandas as pd
study_name = "distributed_dho_hpo"
shared_project_dir = "~/EOB_NN_p4PN/dho_example"

storage_name = f"sqlite:///optuna_optimize_dho.db"
study = optuna.load_study(study_name=study_name, storage=storage_name)
print(study.best_value)