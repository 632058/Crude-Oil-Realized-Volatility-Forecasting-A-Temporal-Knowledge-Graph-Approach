# ============================================================
# GNN Oil Context Embedding Extractor — Notebook Version
# Aligned with get_embedding.py, using variant=full + pooling=mean
# ============================================================

import os


# =========================================================
# [Key change] Force environment variables
# Must be set before importing torch; otherwise PyTorch CUDA
# initialization may occur first and the setting will no longer work.
# =========================================================
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


import warnings
import pickle
import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import random
from pathlib import Path


# Match the import path used in get_embedding.py
from model.model import build_model


warnings.filterwarnings("ignore")



# =========================
# Parameter settings
# =========================
DB_PATH       = "data/gdelt_daily_bilateral_by_eventcode_add.duckdb"
MAPPING_FILE  = "data/gdelt_mappings.pkl"
OIL_CSV       = "data/rv_with_gpr.csv"
OUTPUT_DIR    = "gnn_embeddings"
SYMBOL        = "CL_c1"


SPATIAL_DIM   = 16
TEMPORAL_DIM  = 32
DECODER_DIM   = 96
NHEAD         = 4
NUM_LAYERS    = 1
LR            = 0.0005
WEIGHT_DECAY  = 1e-4


USE_CAMEO_FILTER = True
CAMEO_MIN_ROOT   = 10


# Parameters for model ablation experiments
USE_LAYERNORM      = True
USE_CONTEXT_FUSION = True


WARMUP_START  = "2015-04-01"
WARMUP_END    = "2017-12-31"
WARMUP_EPOCHS = 5
RECORD_START  = "2015-04-01"


VARIANT  = "full"
POOLING  = "mean"


# Embedding output mode:
#   "pooled"    -> Legacy behavior, mean/max pooling over oil-related entities,
#                  output dimension = TEMPORAL_DIM.
#   "node_flat" -> No pooling, output flattened embeddings for each node,
#                  dimension = TEMPORAL_DIM per node.
EMBEDDING_OUTPUT_MODE = "pooled"


# node_flat scope:
#   "oil_entities" -> Output only the nodes tracked by oil_country_codes;
#                     dimension is roughly TEMPORAL_DIM * 45.
#   "all_entities"  -> Output all entities in the mapping, which may be very large.
NODE_FLAT_SCOPE = "all_entities"


SEED     = 42


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# =========================
# Seed
# =========================
def set_seed(seed=42):
    """Fix all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[Warning] Deterministic algorithms not fully supported: {e}")
        
    print(f"Random seed set to: {seed} (Deterministic Mode Enabled)")



# =========================
# Core Learner
# =========================
class RobustOnlineLearner:
    def __init__(
        self,
        db_path,
        mapping_file,
        variant="full",
        spatial_dim=16,
        temporal_dim=32,
        decoder_dim=96,
        nhead=4,
        num_layers=1,
        lr=0.0005,
        weight_decay=1e-4,
        use_cameo_filter=True,
        cameo_min_root=14,
        use_layernorm=True,       
        use_context_fusion=False, 
        device=None,
    ):
        self.db_path          = db_path
        self.spatial_dim      = spatial_dim
        self.temporal_dim     = temporal_dim
        self.decoder_dim      = decoder_dim
        self.use_cameo_filter = use_cameo_filter
        self.cameo_min_root   = cameo_min_root
        self.variant          = variant
        self.device           = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_layernorm    = use_layernorm
        self.use_context_fusion = use_context_fusion


        with open(mapping_file, "rb") as f:
            self.mappings = pickle.load(f)


        self.num_entities = self.mappings["num_entities"]


        self.model = build_model(
            variant=variant,
            num_entities=self.num_entities,
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            decoder_dim=decoder_dim,
            nhead=nhead,
            num_layers=num_layers,
            use_layernorm=self.use_layernorm,
            use_context_fusion=self.use_context_fusion
        ).to(self.device)


        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.PoissonNLLLoss(log_input=True, full=True, reduction="mean")


        self.memory_bank = torch.zeros(self.num_entities, temporal_dim).to(self.device)
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.01)


        self.accum_steps   = 1
        self.current_accum = 0


        self.high_intensity_rel_ids     = self.get_high_intensity_rel_ids(threshold=self.cameo_min_root)
        self.high_intensity_rel_id_set  = set(self.high_intensity_rel_ids)


        print(f"Learner Initialized (Device: {self.device})")
        print(f"Variant: {self.variant}")
        print(f"Dims -> Spatial:{spatial_dim}, Temporal:{temporal_dim}, Decoder:{decoder_dim}")
        print(f"Modules -> LayerNorm: {self.use_layernorm} | Context Fusion: {self.use_context_fusion}")
        print("Architecture: same-day graph update and dyad count reconstruction")
        print(f"CAMEO filter enabled: {self.use_cameo_filter}")
        print("Loss: PoissonNLLLoss on dyad event counts")
        if self.use_cameo_filter:
            print(f"CAMEO root threshold: >= {self.cameo_min_root}")
            print(f"Active relation ids: {len(self.high_intensity_rel_ids)}")


    def reset_memory_bank(self):
        self.memory_bank = torch.zeros(self.num_entities, self.temporal_dim).to(self.device)
        nn.init.normal_(self.memory_bank, mean=0.0, std=0.01)


    def get_high_intensity_rel_ids(self, threshold=10):
        target_ids = []
        all_codes = self.mappings["rel_le"].classes_
        for i, code in enumerate(all_codes):
            try:
                root = int(str(code)[:2])
                if root >= threshold:
                    target_ids.append(i)
            except Exception:
                continue
        return target_ids


    def _apply_cameo_filter(self, df):
        if not self.use_cameo_filter:
            return df
        if not self.high_intensity_rel_id_set:
            return df.iloc[0:0].copy()
        return df[df["rel_id"].isin(self.high_intensity_rel_id_set)].copy()


    def _build_pair_training_frame(self, df):
        pair_df = df.copy()
        pair_df["event_count"] = np.expm1(pair_df["log_weight"].astype(float)).clip(lower=0.0)
        pair_df = pair_df.groupby(["src_id", "dst_id"], as_index=False)["event_count"].sum()
        pair_df["log_event_count"] = np.log1p(pair_df["event_count"])
        return pair_df


    def step(self, time_idx):
        con = duckdb.connect(self.db_path, read_only=True)
        query = f"SELECT src_id, dst_id, rel_id, log_weight FROM training_ready_data WHERE time_idx = {time_idx}"
        df = con.execute(query).df()
        con.close()


        if len(df) == 0:
            self.memory_bank = self.memory_bank.detach()
            return None


        df = self._apply_cameo_filter(df)
        if len(df) == 0:
            self.memory_bank = self.memory_bank.detach()
            return None


        daily_count_val    = np.log1p(len(df))
        daily_count_tensor = torch.tensor([daily_count_val], dtype=torch.float, device=self.device)


        event_src    = torch.tensor(df["src_id"].values,    dtype=torch.long,  device=self.device)
        event_dst    = torch.tensor(df["dst_id"].values,    dtype=torch.long,  device=self.device)
        event_weight = torch.tensor(df["log_weight"].values, dtype=torch.float, device=self.device)
        pos_edge_index = torch.stack([event_src, event_dst], dim=0)


        pair_df    = self._build_pair_training_frame(df)
        pair_src   = torch.tensor(pair_df["src_id"].values,    dtype=torch.long,  device=self.device)
        pair_dst   = torch.tensor(pair_df["dst_id"].values,    dtype=torch.long,  device=self.device)
        pair_target = torch.tensor(pair_df["event_count"].values, dtype=torch.float, device=self.device)


        self.model.train()
        new_memory = self.model.forward_memory_update(
            pos_edge_index,
            event_weight,
            self.memory_bank,
            daily_count_tensor,
        )


        h_s   = new_memory[pair_src]
        h_o   = new_memory[pair_dst]
        preds = self.model.decoder(torch.cat([h_s, h_o], dim=1)).squeeze(-1)
        loss  = self.criterion(preds, pair_target)
        (loss / self.accum_steps).backward()


        with torch.no_grad():
            self.memory_bank = new_memory.detach()


        self.current_accum += 1
        if self.current_accum % self.accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()


        return float(loss.item())


    def get_embedding(self, entity_code=None, entity_id=None):
        self.model.eval()
        with torch.no_grad():
            if entity_code:
                try:
                    eid = self.mappings["entity_le"].transform([entity_code])[0]
                    return self.memory_bank[eid].clone()
                except Exception:
                    return None
            elif entity_id is not None:
                return self.memory_bank[entity_id].clone()
            else:
                return self.memory_bank.clone()


    def get_full_state(self):
        return {
            "model":      self.model.state_dict(),
            "memory":     self.memory_bank.cpu(),
            "optimizer":  self.optimizer.state_dict(),
            "accum_step": self.current_accum,
        }



# =========================
# OilGNNEmbeddingExtractor
# =========================
class OilGNNEmbeddingExtractor(RobustOnlineLearner):
    def __init__(
        self,
        db_path,
        mapping_file,
        variant="full",
        spatial_dim=16,
        temporal_dim=32,
        decoder_dim=64,
        nhead=2,
        num_layers=1,
        lr=0.0005,
        weight_decay=1e-4,
        use_cameo_filter=True,
        cameo_min_root=10,
        use_layernorm=True,       
        use_context_fusion=False, 
        device=None,
    ):
        super().__init__(
            db_path=db_path,
            mapping_file=mapping_file,
            variant=variant,
            spatial_dim=spatial_dim,
            temporal_dim=temporal_dim,
            decoder_dim=decoder_dim,
            nhead=nhead,
            num_layers=num_layers,
            lr=lr,
            weight_decay=weight_decay,
            use_cameo_filter=use_cameo_filter,
            cameo_min_root=cameo_min_root,
            use_layernorm=use_layernorm,            
            use_context_fusion=use_context_fusion,  
            device=device,
        )


        oil_country_codes = [
            "RUS", "SAU", "IRN", "IRQ", "USA", "ARE", "KWT", "QAT", "VEN", "NGA",
            "CHN", "LBN", "GBR", "DEU", "FRA", "IND", "CAN", "BRA",
            "UKR", "TWN", "OMN", "TUR", "EGY", "ISR", "YEM",
            "MEX", "NOR", "LBY", "GUY", "COL", "NLD", "AUS", "SGP", "PAN",
            "DJI", "IDN", "VNM", "PHL", "PSE", "SYR", "PAK",
            "_OPEC", "_IEA", "_TERROR", "_INTL_ORG",
        ]


        self.oil_entity_codes = []
        self.oil_entity_id_list = []
        le = self.mappings["entity_le"]
        for code in oil_country_codes:
            try:
                eid = int(le.transform([code])[0])
                self.oil_entity_codes.append(code)
                self.oil_entity_id_list.append(eid)
            except Exception:
                continue


        self.oil_entity_ids = torch.tensor(
            self.oil_entity_id_list, dtype=torch.long, device=self.device
        )
        print(f"Extractor initialized. Tracking {len(self.oil_entity_ids)} oil-related entities.")


    def warmup_sequential(self, date_to_idx, start_date_str, end_date_str, epochs=4):
        print(f"\nSequential Warmup (Same-day Reconstruction): {start_date_str} to {end_date_str}")


        s_date = pd.to_datetime(start_date_str)
        e_date = pd.to_datetime(end_date_str)


        target_indices = []
        sorted_dates = sorted([d for d in date_to_idx.keys() if s_date <= d <= e_date])
        for d in sorted_dates:
            target_indices.append(date_to_idx[d])


        if not target_indices:
            print("Warning: No data found for warmup period.")
            return


        print(f"Target: {len(target_indices)} days. Running {epochs} sequential loops.")
        self.model.train()


        for epoch in range(epochs):
            self.reset_memory_bank()
            total_loss = 0.0
            count = 0


            pbar = tqdm(target_indices, desc=f"Warmup Loop {epoch + 1}/{epochs}", leave=False)
            for t_idx in pbar:
                loss = self.step(t_idx)
                if loss is not None:
                    total_loss += loss
                    count += 1


            avg_loss = total_loss / count if count > 0 else 0.0
            print(f"Loop {epoch + 1} Done. Avg Loss: {avg_loss:.4f}")


        print("Sequential Warmup Complete. Structure Stabilized.")


    def get_oil_context_vector(self, pooling="mean"):
        """Pool oil-related entity memories into a single TEMPORAL_DIM vector."""
        with torch.no_grad():
            emb = self.memory_bank[self.oil_entity_ids] if self.oil_entity_ids.numel() > 0 else self.memory_bank
            if pooling == "max":
                return emb.max(dim=0).values.cpu().numpy()
            elif pooling == "mean":
                return emb.mean(dim=0).cpu().numpy()
            else:
                raise ValueError(f"Unsupported pooling: {pooling}")


    def get_embedding_feature_dict(self, pooling="mean", output_mode="pooled", node_flat_scope="oil_entities"):
        """Return a single row of embedding features for CSV output."""
        if output_mode == "pooled":
            context_data = self.get_oil_context_vector(pooling=pooling).flatten()
            return {f"gnn_feat_{i}": float(val) for i, val in enumerate(context_data)}


        if output_mode != "node_flat":
            raise ValueError(f"Unsupported EMBEDDING_OUTPUT_MODE: {output_mode}")


        with torch.no_grad():
            if node_flat_scope == "oil_entities":
                if self.oil_entity_ids.numel() > 0:
                    node_ids = self.oil_entity_ids
                    node_labels = self.oil_entity_codes
                    emb = self.memory_bank[node_ids]
                else:
                    node_labels = [f"entity_{i}" for i in range(self.memory_bank.size(0))]
                    emb = self.memory_bank
            elif node_flat_scope == "all_entities":
                node_labels = [str(c) for c in self.mappings["entity_le"].classes_]
                emb = self.memory_bank
            else:
                raise ValueError(f"Unsupported NODE_FLAT_SCOPE: {node_flat_scope}")


            emb_np = emb.cpu().numpy()
            features = {}
            for node_label, node_vec in zip(node_labels, emb_np):
                clean_label = str(node_label).replace("/", "_").replace(" ", "_")
                for j, val in enumerate(node_vec):
                    features[f"gnn_{clean_label}_feat_{j}"] = float(val)
            return features



# =========================
# Helper functions
# =========================
def get_date_to_timeidx_map(mapping_file):
    with open(mapping_file, "rb") as f:
        mappings = pickle.load(f)
    idx_to_date_int = mappings["idx_to_date"]
    return {pd.to_datetime(str(d)): i for i, d in idx_to_date_int.items()}



def make_filter_tag(use_cameo_filter, cameo_min_root):
    return f"cameo_ge{cameo_min_root}" if use_cameo_filter else "all_relations"



# =========================
# Main execution logic
# =========================
def run_embedding_extraction():
    set_seed(SEED)


    date_to_idx  = get_date_to_timeidx_map(MAPPING_FILE)
    max_gnn_date = max(date_to_idx.keys())
    print(f"GNN DB Max Date    : {max_gnn_date.date()} (Will strictly truncate here)")


    oil_df         = pd.read_csv(OIL_CSV)
    oil_df["date"] = pd.to_datetime(oil_df["date"])
    start_cutoff   = pd.to_datetime(RECORD_START)
    oil_df         = oil_df[oil_df["date"] >= start_cutoff]
    oil_df         = oil_df[oil_df["symbol"] == SYMBOL].dropna(subset=["rv"])
    all_dates      = sorted(oil_df["date"].unique())


    if not all_dates:
        print("Error: No valid data found in OIL_CSV after filtering.")
        return None


    print(f"Data range detected: {all_dates[0].date()} to {all_dates[-1].date()}")
    print(f"Variant: {VARIANT} | Pooling: {POOLING}")


    extractor = OilGNNEmbeddingExtractor(
        db_path=DB_PATH,
        mapping_file=MAPPING_FILE,
        variant=VARIANT,
        spatial_dim=SPATIAL_DIM,
        temporal_dim=TEMPORAL_DIM,
        decoder_dim=DECODER_DIM,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        use_cameo_filter=USE_CAMEO_FILTER,
        cameo_min_root=CAMEO_MIN_ROOT,
        use_layernorm=USE_LAYERNORM,             
        use_context_fusion=USE_CONTEXT_FUSION,   
        device=DEVICE,
    )


    # Phase 1: Warmup
    extractor.warmup_sequential(
        date_to_idx,
        start_date_str=WARMUP_START,
        end_date_str=WARMUP_END,
        epochs=WARMUP_EPOCHS,
    )


    # Phase 2: Reset memory so the trained weights can be rerun for burn-in and OOS
    print("\nReset memory after warmup. Burn-in/OOS pass will restart from record_start with trained model weights.")
    extractor.reset_memory_bank()


    oil_context_list = []


    print("\n=== Start extracting GNN embedding history (Burn-in & OOS) ===")
    print(f"Warmup training only      : {WARMUP_START} -> {WARMUP_END}")
    print(f"Burn-in/recording starts  : {start_cutoff.date()} -> {max_gnn_date.date()}")
    print(f"Embedding output starts   : {start_cutoff.date()}")
    print(f"Variant                  : {VARIANT}")
    print(f"Pooling                  : {POOLING}")
    print(f"Embedding output mode    : {EMBEDDING_OUTPUT_MODE}")
    if EMBEDDING_OUTPUT_MODE == "node_flat":
        if NODE_FLAT_SCOPE == "oil_entities":
            expected_dim = TEMPORAL_DIM * len(extractor.oil_entity_id_list)
        elif NODE_FLAT_SCOPE == "all_entities":
            expected_dim = TEMPORAL_DIM * extractor.num_entities
        else:
            expected_dim = None
        print(f"Node flat scope          : {NODE_FLAT_SCOPE}")
        print(f"Output dimension per day : {expected_dim}")
    else:
        print(f"Output dimension per day : {TEMPORAL_DIM}")
    print(f"CAMEO filter enabled     : {USE_CAMEO_FILTER}")
    if USE_CAMEO_FILTER:
        print(f"CAMEO root threshold     : >= {CAMEO_MIN_ROOT}")


    for current_date in tqdm(all_dates, desc=f"Extracting raw embeddings [{VARIANT}]..."):
        if current_date > max_gnn_date:
            print(f"\n[Info] Reached GNN data limit ({max_gnn_date.date()}). Truncating extraction.")
            break


        if current_date in date_to_idx:
            t_idx = date_to_idx[current_date]
            try:
                extractor.step(t_idx)
            except Exception as e:
                print(f"\n[CRITICAL ERROR] Failed to process date {current_date.date()} (t_idx: {t_idx}). Reason: {e}")


        if current_date >= start_cutoff:
            entry = {"date": current_date}
            entry.update(
                extractor.get_embedding_feature_dict(
                    pooling=POOLING,
                    output_mode=EMBEDDING_OUTPUT_MODE,
                    node_flat_scope=NODE_FLAT_SCOPE,
                )
            )
            oil_context_list.append(entry)


    # Output
    embedding_df = pd.DataFrame(oil_context_list)
    filter_tag   = make_filter_tag(USE_CAMEO_FILTER, CAMEO_MIN_ROOT)
    output_dir   = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)


    mode_tag = EMBEDDING_OUTPUT_MODE
    if EMBEDDING_OUTPUT_MODE == "node_flat":
        mode_tag = f"nodeflat_{NODE_FLAT_SCOPE}"
    else:
        mode_tag = POOLING


    ablation_tag = f"norm{'T' if USE_LAYERNORM else 'F'}_fus{'T' if USE_CONTEXT_FUSION else 'F'}"
    stem         = f"{VARIANT}_{mode_tag}_{ablation_tag}_from_{pd.to_datetime(RECORD_START).strftime('%Y%m%d')}_{filter_tag}"
    csv_filename = output_dir / f"gnn_oil_context_{stem}.csv"


    embedding_df.to_csv(csv_filename, index=False)


    print(f"\nExtraction Task Completed.")
    if not embedding_df.empty:
        print(f"[{VARIANT} | {POOLING}] rows: {len(embedding_df)}")
        print(f"[{VARIANT} | {POOLING}] Start Date: {embedding_df['date'].min().date()}")
        print(f"[{VARIANT} | {POOLING}] End Date  : {embedding_df['date'].max().date()}")
        print(f"[{VARIANT} | {POOLING}] Embedding CSV: {csv_filename}")


    return embedding_df



# =========================
# Entry point
# =========================
if __name__ == "__main__":
    history_df = run_embedding_extraction()