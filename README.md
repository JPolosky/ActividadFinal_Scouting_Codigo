# Sistema de Scouting Deportivo – Clasificación de Jugadores Élite

Ecosistema analítico de extremo a extremo que utiliza **Machine Learning** para estimar si un jugador de fútbol pertenece al grupo **Élite** o **Estándar**, según su valor de mercado. El proyecto integra extracción de datos desde MySQL, preprocesamiento, entrenamiento de un modelo Random Forest, una API REST desarrollada con FastAPI y una interfaz web.

> Proyecto académico – Electiva II, Ingeniería de Sistemas  
> Universidad Cooperativa de Colombia  
> Autores: Manuel Pacheco y Joel Polo  
> Fecha: noviembre de 2026

## 1. Objetivo

Construir una herramienta de apoyo al scouting que permita priorizar jugadores para una evaluación más detallada. El modelo recibe edad, estatura y posición, y devuelve una predicción de clase junto con la probabilidad estimada de pertenecer al grupo Élite.

La etiqueta se construye a partir del percentil 75 del valor de mercado: los jugadores cuyo valor es igual o superior a ese umbral se consideran Élite. En los datos utilizados en el proyecto, el umbral fue de **700.000 EUR**.

**Importante:** la predicción es un filtro de apoyo y no debe interpretarse como una recomendación automática de fichaje. El modelo utiliza un conjunto limitado de variables y requiere validación por parte de ojeadores.

## 2. Características principales

- Extracción de jugadores desde MySQL mediante SQLAlchemy y PyMySQL.
- Preparación de datos, imputación de estatura por mediana de posición y creación de la variable objetivo.
- Codificación One-Hot de la posición.
- Comparación de árboles de decisión para estudiar el sobreajuste.
- Entrenamiento y optimización de Random Forest mediante `GridSearchCV`.
- Evaluación con accuracy, precision, recall, F1 y ROC-AUC.
- Serialización del modelo y sus parámetros de preprocesamiento con Joblib.
- API REST con FastAPI y validación de entradas mediante Pydantic.
- Interfaz web que consume la API y presenta la predicción.

## 3. Estructura del proyecto

```text
.
├── index.html                         # Interfaz web
├── main_api.py                        # API REST con FastAPI
├── exportar_modelo.py                 # Entrenamiento, evaluación y exportación
├── models/
│   └── modelo_scouting_rf.joblib      # Modelo entrenado y metadatos
└── src/
    └── database/
        └── connection2.py             # Conector a MySQL
```

Al ejecutar el entrenamiento se generan también los directorios `reports/` y `reports/figures/`, junto con el archivo `reports/metricas.json`.

## 4. Requisitos

- Python 3.10 o superior (se recomienda utilizar un entorno virtual).
- MySQL accesible desde el equipo donde se ejecutará el entrenamiento.
- Base de datos con la tabla `players` y los campos utilizados por la consulta del proyecto:
  - `player_id`
  - `name`
  - `position`
  - `height_in_cm`
  - `date_of_birth`
  - `market_value_in_eur`
- Git (opcional, para clonar y versionar el repositorio).

## 5. Instalación

Clona el repositorio y entra en su carpeta:

```bash
git clone <URL_DEL_REPOSITORIO>
cd <CARPETA_DEL_REPOSITORIO>
```

Crea y activa un entorno virtual.

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / Fedora:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instala las dependencias:

```bash
python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" pandas numpy scikit-learn joblib matplotlib sqlalchemy pymysql
```

## 6. Configuración de la base de datos

El conector `src/database/connection2.py` obtiene los datos de conexión desde variables de entorno. Configura estos valores para tu instalación:

| Variable | Descripción | Valor local predeterminado del código |
|---|---|---|
| `MYSQL_HOST` | Servidor MySQL | `127.0.0.1` |
| `MYSQL_PORT` | Puerto MySQL | `3307` |
| `MYSQL_USER` | Usuario de MySQL | `scouting` |
| `MYSQL_PASSWORD` | Contraseña de MySQL | No utilizar el valor de ejemplo en producción |
| `MYSQL_DB` | Nombre de la base de datos | `scouting_db` |

**PowerShell (solo para la sesión actual):**

```powershell
$env:MYSQL_HOST = "127.0.0.1"
$env:MYSQL_PORT = "3307"
$env:MYSQL_USER = "tu_usuario"
$env:MYSQL_PASSWORD = "tu_contraseña"
$env:MYSQL_DB = "scouting_db"
```

**Linux / Fedora (solo para la sesión actual):**

```bash
export MYSQL_HOST="127.0.0.1"
export MYSQL_PORT="3307"
export MYSQL_USER="tu_usuario"
export MYSQL_PASSWORD="tu_contraseña"
export MYSQL_DB="scouting_db"
```

No publiques contraseñas, archivos `.env`, tokens ni credenciales en el repositorio. Si se utiliza un archivo `.env`, debe estar excluido mediante `.gitignore` y cargarse explícitamente con una herramienta como `python-dotenv` (esa carga no está implementada actualmente en el código entregado).

Para comprobar la conexión y consultar el número de jugadores:

```bash
python src/database/connection2.py
```

El entrenamiento consulta la tabla `players` y requiere que la base de datos esté disponible.

## 7. Entrenamiento del modelo

Desde la raíz del proyecto, con el entorno virtual activo y MySQL configurado:

```bash
python exportar_modelo.py
```

El proceso extrae los datos, calcula el umbral del percentil 75, prepara las variables, compara modelos, optimiza el Random Forest y guarda:

- `models/modelo_scouting_rf.joblib`: modelo entrenado, columnas esperadas, medianas de imputación, umbral y métricas.
- `reports/metricas.json`: resumen de datos, experimentos y evaluación.
- `reports/figures/`: visualizaciones generadas durante el análisis.

La API necesita que exista `models/modelo_scouting_rf.joblib`. El ZIP de código entregado ya incluye un modelo serializado, por lo que se puede probar la API sin reentrenar, siempre que se conserven las versiones compatibles de las librerías.

## 8. Ejecutar la API y la aplicación web

Desde la raíz del proyecto:

```bash
uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

Abre en el navegador:

- Aplicación web: <http://127.0.0.1:8000/>
- Documentación interactiva: <http://127.0.0.1:8000/docs>
- Estado del servicio: <http://127.0.0.1:8000/health>
- Información del modelo: <http://127.0.0.1:8000/modelo/info>

Para permitir conexiones desde otros equipos de la red local, puede ejecutarse con `--host 0.0.0.0`; en ese caso se debe configurar el firewall y utilizar la IP del equipo servidor. No expongas el servicio a Internet sin autenticación, controles de acceso y configuración de seguridad adecuados.

## 9. Endpoints de la API

### `GET /health`

Comprueba que la API está activa y que el modelo fue cargado.

### `GET /modelo/info`

Devuelve la versión, fecha de entrenamiento, umbral de valor de mercado, métricas de prueba, mejores hiperparámetros y columnas de entrada.

### `POST /predict`

Recibe los datos de un jugador y devuelve su clasificación y probabilidad estimada.

Ejemplo de solicitud JSON:

```json
{
  "jugador": "Candidato A",
  "posicion": "Defender",
  "estatura_cm": 188,
  "edad": 22
}
```

Valores admitidos para `posicion`:

- `Attack`
- `Defender`
- `Goalkeeper`
- `Midfield`
- `Missing`

Los campos `estatura_cm` y `edad` son opcionales. Si no se envía la estatura, la API utiliza la mediana guardada para la posición; si no se envía la edad, utiliza la mediana global guardada en el paquete del modelo.

Ejemplo de respuesta:

```json
{
  "jugador": "Candidato A",
  "prediccion": 1,
  "etiqueta": "Élite",
  "probabilidad_elite": 64.7,
  "umbral_valor_eur": 700000.0,
  "campos_imputados": []
}
```

La respuesta real depende del modelo cargado y de los datos enviados.

## 10. Resultados de referencia

Según el informe del proyecto, el Random Forest optimizado obtuvo en el conjunto de prueba:

| Métrica | Resultado |
|---|---:|
| Accuracy | 60,7 % |
| Precision | 38,2 % |
| Recall | 76,3 % |
| F1 | 50,9 % |
| ROC-AUC | 0,694 |

La edad fue la variable de mayor importancia en el modelo. Estos resultados corresponden al experimento documentado y pueden cambiar si se modifica la base de datos, el preprocesamiento o los hiperparámetros.

## 11. Solución de problemas

**La API indica que no encuentra el modelo**

Comprueba que exista `models/modelo_scouting_rf.joblib`. Si no existe, ejecuta `python exportar_modelo.py` después de configurar MySQL.

**No se puede conectar a MySQL**

Verifica que el servicio esté activo, que el host y el puerto sean correctos, que la base de datos exista y que el usuario tenga permisos de lectura sobre `players`.

**`uvicorn` no se reconoce**

Activa el entorno virtual e instala las dependencias indicadas en la sección de instalación. También puedes iniciar con:

```bash
python -m uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
```

**La página no obtiene predicciones**

Confirma que la API esté ejecutándose y que `http://127.0.0.1:8000/health` responda correctamente. La interfaz permite configurar la URL base de la API.

## 12. Tecnologías utilizadas

- Python
- pandas y NumPy
- scikit-learn
- Random Forest y Decision Tree
- Matplotlib
- Joblib
- MySQL, SQLAlchemy y PyMySQL
- FastAPI, Pydantic y Uvicorn
- HTML, CSS y JavaScript

## 13. Alcance y trabajo futuro

El modelo utiliza edad, estatura y posición. Para mejorar su capacidad predictiva se recomienda incorporar variables deportivas y contextuales, como goles, asistencias, minutos jugados, liga, club y rendimiento por temporada. También se recomienda entrenar con jugadores activos, revisar periódicamente el umbral de clasificación y evaluar el modelo con datos de temporadas posteriores.

---

**Universidad Cooperativa de Colombia – Ingeniería de Sistemas**  
**Electiva II – Actividad final (Momento 3)**
