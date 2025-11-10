
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Literal
from intent_drift_class import IntentDriftEngine, Constraint

app = FastAPI(title="Intent Drift Service", version="1.0.0", description="Stores desired intent and computes drift from posted actuals.")

# --- Pydantic schemas ---
class ConstraintIn(BaseModel):
    name: str = Field(..., description="Feature name, e.g., 'latency_ms'")
    op: Literal["<=", ">=", "=="] = Field(..., description="Constraint operator")
    target: float = Field(..., description="Target value")
    weight: float = Field(1.0, ge=0.0, description="Importance weight for aggregation")
    norm: Optional[float] = Field(None, description="Normalization factor for this feature")

class ConstraintsSetRequest(BaseModel):
    constraints: List[ConstraintIn]

class ConstraintsResponse(BaseModel):
    constraints: List[ConstraintIn]

class ActualsIn(BaseModel):
    actuals: Dict[str, float]
    metric: Literal["l2", "cosine"] = "l2"
    theta: Optional[float] = Field(None, description="Threshold to flag drift")

    @validator("actuals")
    def non_empty_actuals(cls, v):
        if not v:
            raise ValueError("actuals must not be empty")
        return v

class DriftResult(BaseModel):
    metric: str
    drift: float
    threshold: Optional[float]
    is_drift: bool
    intent_vector: List[float]
    state_vector: List[float]
    feature_names: List[str]
    violations: List[Dict]

# --- Engine state (in-memory) ---
engine = IntentDriftEngine()

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/constraints", response_model=ConstraintsResponse)
def get_constraints():
    return {"constraints": [ConstraintIn(**c.__dict__) for c in engine.constraints]}

@app.post("/constraints", response_model=ConstraintsResponse)
def set_constraints(req: ConstraintsSetRequest):
    engine.set_constraints([Constraint(**c.dict()) for c in req.constraints])
    return {"constraints": req.constraints}

@app.post("/drift", response_model=DriftResult)
def compute_drift(req: ActualsIn):
    try:
        res = engine.compute(actuals=req.actuals, metric=req.metric, theta=req.theta)
        return DriftResult(**res)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

# Convenience route to add/append a single constraint
@app.post("/constraints/add", response_model=ConstraintsResponse)
def add_constraint(c: ConstraintIn):
    engine.add_constraint(Constraint(**c.dict()))
    return {"constraints": [ConstraintIn(**c.__dict__) for c in engine.constraints]}
