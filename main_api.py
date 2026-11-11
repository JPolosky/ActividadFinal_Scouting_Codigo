"""
API del modelo de scouting hecha con FastAPI.

Para ejecutarla:
    uvicorn main_api:app --reload --port 8000

Rutas:
    GET  /health       -> estado de la API
    GET  /modelo/info  -> datos del modelo (métricas, umbral, etc.)
    POST /predict      -> recibe un jugador y devuelve si es Élite y la probabilidad
    GET  /docs         -> documentación de FastAPI
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

RUTA_MODELO = Path(__file__).resolve().parent / "models" / "modelo_scouting_rf.joblib"

try:
    PAQUETE = joblib.load(RUTA_MODELO)
except FileNotFoundError as exc:
    raise RuntimeError(f"No se encontró el modelo en {RUTA_MODELO}. Ejecute exportar_modelo.py") from exc

MODELO = PAQUETE["modelo"]
COLUMNAS = PAQUETE["columnas_X"]


class Jugador(BaseModel):
    # estatura y edad son opcionales: si no llegan se llenan con las medianas
    # guardadas en el paquete (igual que en el entrenamiento)
    model_config = ConfigDict(json_schema_extra={"example": {
        "jugador": "Candidato A", "posicion": "Defender", "estatura_cm": 188, "edad": 22}})

    jugador: str | None = Field(None, max_length=100)
    posicion: Literal["Attack", "Defender", "Goalkeeper", "Midfield", "Missing"]
    estatura_cm: float | None = Field(None, ge=150, le=215)
    edad: int | None = Field(None, ge=15, le=50)


class Prediccion(BaseModel):
    jugador: str | None
    prediccion: int = Field(..., description="1 = Élite, 0 = Estándar")
    etiqueta: str
    probabilidad_elite: float = Field(..., description="Probabilidad de ser Élite en %")
    umbral_valor_eur: float
    campos_imputados: list[str]


app = FastAPI(title="API Scouting (Random Forest)", version=PAQUETE.get("version", "1.0.0"),
              description="Clasifica jugadores como Élite (top 25 % de valor de mercado) o Estándar.")
# CORS para que index.html funcione aunque se abra como archivo
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def preparar(jugador):
    datos = jugador.model_dump()
    imputados = []
    if datos["estatura_cm"] is None:
        datos["estatura_cm"] = PAQUETE["medianas_estatura"].get(datos["posicion"],
                                                                PAQUETE["mediana_global_estatura"])
        imputados.append("estatura_cm")
    if datos["edad"] is None:
        datos["edad"] = PAQUETE["mediana_global_edad"]
        imputados.append("edad")
    df = pd.DataFrame([datos])
    # sin drop_first: con un solo jugador eliminaría justo su posición.
    # reindex deja solo las columnas del entrenamiento (la de Attack se descarta sola)
    dummies = pd.get_dummies(df[["posicion"]], dtype=int)
    X = pd.concat([df[["edad", "estatura_cm"]], dummies], axis=1)
    return X.reindex(columns=COLUMNAS, fill_value=0), imputados


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": True, "version": PAQUETE.get("version")}


@app.get("/modelo/info")
def info():
    return {"version": PAQUETE.get("version"), "entrenado": PAQUETE.get("entrenado"),
            "umbral_valor_eur": PAQUETE["umbral_elite_eur"], "metricas_test": PAQUETE["metricas_test"],
            "mejores_parametros": PAQUETE["mejores_parametros"], "columnas_X": COLUMNAS}


@app.post("/predict", response_model=Prediccion)
def predict(jugador: Jugador):
    try:
        X, imputados = preparar(jugador)
        proba = float(MODELO.predict_proba(X)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al generar la predicción: {exc}") from exc
    pred = int(proba >= 0.5)
    return Prediccion(jugador=jugador.jugador, prediccion=pred, etiqueta="Élite" if pred else "Estándar",
                      probabilidad_elite=round(proba * 100, 2),
                      umbral_valor_eur=PAQUETE["umbral_elite_eur"], campos_imputados=imputados)


@app.get("/", include_in_schema=False)
def cliente_web():
    # la página web también se puede abrir directamente
    return FileResponse(Path(__file__).resolve().parent / "index.html")
