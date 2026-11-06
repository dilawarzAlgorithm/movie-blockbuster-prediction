from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
import uvicorn

from predict import predict_blockbuster

app = FastAPI(
    title="Movie Blockbuster Prediction API",
    description="Predicts if an upcoming movie will be a commercial blockbuster before release.",
    version="1.0.0",
)


class MovieInput(BaseModel):
    year: int = Field(..., description="Release year")
    decade: int = Field(..., description="Release decade")
    genre: str = Field(..., description="Movie genre")
    runtime_min: float = Field(..., gt=0, description="Runtime in minutes")
    budget_usd: float = Field(..., ge=0, description="Production budget in USD")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "year": 2024,
                "decade": 2020,
                "genre": "Sci-Fi",
                "runtime_min": 140.0,
                "budget_usd": 180000000.0,
            }
        }
    )


class PredictionOutput(BaseModel):
    blockbuster_probability: float
    is_blockbuster: bool


@app.get("/health", status_code=200)
def health():
    return {"status": "OK"}


@app.post("/predict", response_model=PredictionOutput)
def predict(movie: MovieInput):
    try:
        movie_dict = movie.model_dump()
        result = predict_blockbuster(movie_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)