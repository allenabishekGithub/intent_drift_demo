
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class Constraint:
    """Represents a single SLO/SLA constraint."""
    name: str
    op: str          # one of {"<=", ">=", "=="}
    target: float
    weight: float = 1.0
    norm: Optional[float] = None  # per-feature normalization (e.g., target or max)

class IntentDriftEngine:
    """Class-based engine for computing intent drift between desired and actual network states.
    
    Functionality mirrors the functional implementation:
      - Weighted L2 or cosine distance on normalized feature vectors
      - Per-constraint directional violations
      - Optional thresholding to flag drift
      - KL divergence helper for distributional drift
    """
    def __init__(self, constraints: Optional[List[Constraint]] = None):
        self.constraints: List[Constraint] = constraints or []

    # -------- Constraint management --------
    def set_constraints(self, constraints: List[Constraint]) -> None:
        self.constraints = list(constraints)

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def clear_constraints(self) -> None:
        self.constraints.clear()

    # -------- Core helpers (private) --------
    def _vectorize(self, actuals: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        names, I, S, W = [], [], [], []
        for c in self.constraints:
            if c.name not in actuals:
                raise KeyError(f"Actuals missing feature '{c.name}'")
            names.append(c.name)
            I.append(c.target); S.append(actuals[c.name]); W.append(c.weight)
        return np.array(I, float), np.array(S, float), np.array(W, float), names

    @staticmethod
    def _normalize(I: np.ndarray, S: np.ndarray, norms: List[Optional[float]]):
        if norms is None:
            return I, S
        I2, S2 = I.copy(), S.copy()
        for i, n in enumerate(norms):
            if n and n != 0:
                I2[i] /= n; S2[i] /= n
        return I2, S2

    @staticmethod
    def _directional_error(op: str, target: float, actual: float) -> float:
        if op == "<=":
            return max(0.0, actual - target)
        elif op == ">=":
            return max(0.0, target - actual)
        elif op == "==":
            return abs(actual - target)
        else:
            raise ValueError(f"Unsupported operator: {op}")

    # -------- Distance metrics --------
    @staticmethod
    def l2_drift(I: np.ndarray, S: np.ndarray, W: Optional[np.ndarray] = None) -> float:
        diff = S - I
        return float(np.sqrt(np.sum((W * diff**2) if W is not None else diff**2)))

    @staticmethod
    def cosine_drift(I: np.ndarray, S: np.ndarray) -> float:
        denom = (np.linalg.norm(I) * np.linalg.norm(S))
        return 0.0 if denom == 0 else float(1.0 - np.dot(I, S) / denom)

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
        p = np.clip(p, eps, None); q = np.clip(q, eps, None)
        p = p / p.sum(); q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    # -------- Public API --------
    def compute(self, actuals: Dict[str, float], metric: str = "l2", theta: Optional[float] = None) -> Dict:
        """Compute intent drift using the current constraints and supplied actuals."""
        if not self.constraints:
            raise ValueError("No constraints set. Use set_constraints() or add_constraint().")

        I, S, W, names = self._vectorize(actuals)
        norms = [c.norm for c in self.constraints]
        I_n, S_n = self._normalize(I, S, norms)

        if metric == "l2":
            drift_value = self.l2_drift(I_n, S_n, W)
        elif metric == "cosine":
            drift_value = self.cosine_drift(I_n, S_n)
        else:
            raise ValueError("Unsupported metric. Use 'l2' or 'cosine'.")

        violations = []
        for c in self.constraints:
            raw_err = self._directional_error(c.op, c.target, actuals[c.name])
            err = raw_err / (c.norm if c.norm and c.norm != 0 else 1.0)
            violations.append({
                "feature": c.name,
                "operator": c.op,
                "target": c.target,
                "actual": actuals[c.name],
                "norm": c.norm if c.norm else 1.0,
                "weight": c.weight,
                "normalized_error": err,
                "violates": bool(raw_err > 0) if c.op in ("<=", ">=") else bool(raw_err > 0),
            })

        return {
            "metric": metric,
            "drift": drift_value,
            "threshold": theta,
            "is_drift": (theta is not None and drift_value > theta),
            "intent_vector": I_n.tolist(),
            "state_vector": S_n.tolist(),
            "feature_names": names,
            "violations": violations,
        }
