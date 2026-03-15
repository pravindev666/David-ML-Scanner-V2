"""
DAVID PROPHETIC ORACLE — 5-State Regime Detector
===================================================
Gaussian HMM with 5 states for granular market regime classification.
Also computes Markov transition probabilities: "what regime is next?"
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    raise ImportError("hmmlearn required. Install: pip install hmmlearn")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, REGIME_LABELS, C


class RegimeDetector:
    """
    5-State Gaussian HMM for Market Regime Detection.
    
    States:
        0: Strong Bullish  (high positive returns, low vol)
        1: Mild Bullish    (moderate positive returns)
        2: Sideways        (low returns, low vol — range)
        3: Mild Bearish    (moderate negative returns)
        4: Strong Bearish  (high negative returns, high vol)
    
    Also reports micro-direction probabilities within each regime.
    """
    
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.hmm = None
        self.scaler = StandardScaler()
        self.regime_map = {}  # Maps HMM state index → human label
        self.transition_matrix = None
        self.is_trained = False
        self.hmm_features = [
            "returns_1d", "realized_vol_10", "vol_of_vol",
            "rsi_7", "bb_width", "adx", "vix_percentile"
        ]
    
    def train(self, df, verbose=True):
        """
        Train the 5-state HMM on the full historical dataset.
        Auto-labels states based on the average return of each state.
        """
        # Select features for HMM
        available = [f for f in self.hmm_features if f in df.columns]
        if len(available) < 5:
            raise ValueError(f"Not enough features for HMM. Need at least 5, got {len(available)}")
        
        X = df[available].values
        X_scaled = self.scaler.fit_transform(X)
        
        if verbose:
            print(f"\n{C.header('TRAINING 5-STATE HMM REGIME DETECTOR')}")
            print(f"  Features: {available}")
            print(f"  Samples: {len(X_scaled)}")
        
        # Train HMM with multiple restarts for robustness
        best_score = -np.inf
        best_model = None
        
        for i in range(5):
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=200,
                    random_state=42 + i,
                    tol=0.01,
                )
                model.fit(X_scaled)
                score = model.score(X_scaled)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue
        
        if best_model is None:
            raise RuntimeError("HMM training failed after 5 attempts")
        
        self.hmm = best_model
        
        # Auto-label states based on average returns in each state
        states = self.hmm.predict(X_scaled)
        df_temp = df.copy()
        df_temp["hmm_state"] = states
        
        state_stats = df_temp.groupby("hmm_state").agg({
            "returns_1d": ["mean", "std"],
        }).reset_index()
        state_stats.columns = ["state", "mean_return", "std_return"]
        
        # Sort states by mean return (most bearish → most bullish)
        sorted_states = state_stats.sort_values("mean_return").reset_index(drop=True)
        
        # Map: lowest return → Strong Bearish(4), highest → Strong Bullish(0)
        label_order = [4, 3, 2, 1, 0]  # Strong Bear → Strong Bull
        self.regime_map = {}
        for idx, row in sorted_states.iterrows():
            if idx < len(label_order):
                self.regime_map[int(row["state"])] = REGIME_LABELS[label_order[idx]]
        
        # Compute transition matrix
        self.transition_matrix = self.hmm.transmat_
        
        self.is_trained = True
        
        if verbose:
            print(f"\n  {C.GREEN}[OK] HMM Trained. Log-likelihood: {best_score:.1f}{C.RESET}")
            print(f"\n  State Labels (auto-detected):")
            for state, label in sorted(self.regime_map.items()):
                stats = state_stats[state_stats["state"] == state].iloc[0]
                print(f"    State {state} → {label:>16} | Avg Return: {stats['mean_return']*100:+.3f}%")
        
        return states
    
    def get_current_regime(self, df):
        """
        Predict the regime for the latest observation.
        Returns: (regime_label, state_index, state_probabilities)
        """
        if not self.is_trained:
            return "UNKNOWN", -1, {}
        
        available = [f for f in self.hmm_features if f in df.columns]
        last_obs = df[available].iloc[-1:].values
        last_scaled = self.scaler.transform(last_obs)
        
        state = self.hmm.predict(last_scaled)[0]
        
        # Get posterior probabilities for all states
        log_prob, posteriors = self.hmm.score_samples(last_scaled)
        state_probs = posteriors[0]
        
        label = self.regime_map.get(state, "UNKNOWN")
        
        # Build human-readable probability dict
        prob_dict = {}
        for s, lbl in self.regime_map.items():
            if s < len(state_probs):
                prob_dict[lbl] = float(state_probs[s])
        
        return label, state, prob_dict
    
    def get_transition_probabilities(self, current_state):
        """
        Given current state, return the probabilities of transitioning to each regime.
        "What is the chance of switching to Bearish tomorrow?"
        """
        if self.transition_matrix is None:
            return {}
        
        if current_state >= self.transition_matrix.shape[0]:
            return {}
        
        trans_probs = self.transition_matrix[current_state]
        result = {}
        for s, label in self.regime_map.items():
            if s < len(trans_probs):
                result[label] = float(trans_probs[s])
        
        return result
    
    def get_regime_with_micro_direction(self, df, ensemble_pred=None):
        """
        Returns the regime PLUS absolute micro-direction probability.
        Even in "SIDEWAYS", reports "55% lean UP".
        
        Uses the ensemble classifier's UP/DOWN probabilities to determine
        micro-direction within the regime.
        """
        label, state_idx, state_probs = self.get_current_regime(df)
        trans_probs = self.get_transition_probabilities(state_idx)
        
        # Micro-direction from ensemble
        micro = "FLAT"
        micro_pct = 50.0
        if ensemble_pred:
            p_up = ensemble_pred.get("prob_up", 0.33)
            p_down = ensemble_pred.get("prob_down", 0.33)
            if p_up > p_down:
                micro = "LEAN UP"
                micro_pct = p_up * 100
            elif p_down > p_up:
                micro = "LEAN DOWN"
                micro_pct = p_down * 100
        
        return {
            "regime": label,
            "state_idx": state_idx,
            "state_probs": state_probs,
            "transition_probs": trans_probs,
            "micro_direction": micro,
            "micro_pct": micro_pct,
        }
    
    def save(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "regime_detector.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "hmm": self.hmm,
                "scaler": self.scaler,
                "regime_map": self.regime_map,
                "transition_matrix": self.transition_matrix,
                "hmm_features": self.hmm_features,
            }, f)
        print(f"  {C.GREEN}[SAVED] Regime detector → {path}{C.RESET}")
    
    def load(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "regime_detector.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.hmm = data["hmm"]
        self.scaler = data["scaler"]
        self.regime_map = data["regime_map"]
        self.transition_matrix = data["transition_matrix"]
        self.hmm_features = data.get("hmm_features", self.hmm_features)
        self.is_trained = True
        print(f"  {C.GREEN}[LOADED] Regime detector from {path}{C.RESET}")
        return True
