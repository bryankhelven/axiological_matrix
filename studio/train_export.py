"""Train and serialize verified linear baselines using original released gold."""
import sys
import pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"studio"))
from core import train,save_bundle,load_bundle,analyze
models,specs=train(ROOT)
output=ROOT/"studio"/"model_bundle"
save_bundle(output,models,specs)
loaded,loaded_specs=load_bundle(output)
test=["Precisamos de educação crítica sobre IA.",
      "Ontem choveu muito no meu bairro.",""]
pred=analyze(test,loaded,loaded_specs)
assert len(pred)==3
assert pd.isna(pred.loc[2,"acceptance_3"])
print("TRAINING_INPUTS", {k:v["train_n"] for k,v in specs.items()})
print(pred.to_string(index=False))
print("SERIALIZATION_OK")
