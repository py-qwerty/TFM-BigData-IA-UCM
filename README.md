# TFM-BigData-IA-UCM

Código del TFM: integración de **IA** para una academia de preparación a **Policía Nacional**.  
Incluye tres módulos independientes: **RAG legal**, **clasificador jerárquico** y **planificador de estudio**.

---

## Estructura
```
(1)_rag_project/                 # RAG + API (FastAPI), sincronización BOE, SQL utils
  ├─ deployment/
  ├─ notebooks/
  ├─ routes/
  ├─ sql_files/
  ├─ utils/
  └─ requirements.txt
(2)_category_chapter_classifier/ # Clasificador por bloque y por tema (jerárquico)
  ├─ models_hierarchical/
  ├─ notebooks/
  └─ requirements.txt
(3)_planificador/                # Optimizador de horas por tema
  ├─ optimizer.py
  ├─ testing.ipynb
  └─ requirements.txt
README.md
```

---

## Requisitos
- Python 3.10+ (recomendado 3.11)
- Cuenta/clave de OpenAI (embeddings/LLM)
- Postgres/Supabase (si usas el stack completo del RAG)

---

## Instalación rápida
```bash
python -m venv .venv && source .venv/bin/activate  # (Win) .venv\Scripts\activate
pip install -r "(1)_rag_project/requirements.txt"
pip install -r "(2)_category_chapter_classifier/requirements.txt"
pip install -r "(3)_planificador/requirements.txt"
```

Crea un `.env` (en la raíz o en cada módulo según tu preferencia):
```env
OPENAI_API_KEY=sk-xxxxxxxx
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_ANON_KEY=xxxxxxxx
DATABASE_URL=postgresql://user:pass@host:5432/db
```

---

## Módulos

### (1) RAG legal (BOE)
- **Qué hace**: indexa normativa consolidada, actualiza cambios y expone un **API** para consulta con citas legales.
- **Stack**: FastAPI + Postgres/Supabase + embeddings.
- **Cómo ejecutar (local)**:
  ```bash
  cd "(1)_rag_project"
  uvicorn routes.api:app --reload --port 8000
  ```
- **Docs**: Swagger en `http://localhost:8000/docs` (FastAPI).
- **Nota**: los scripts SQL están en `sql_files/`.

### (2) Clasificador jerárquico
- **Qué hace**: predice **bloque** (Derecho/Sociología/Científico-Técnico) y **tema (1–45)** usando embeddings + SVM jerárquica.
- **Entrenamiento/uso**:
  ```bash
  cd "(2)_category_chapter_classifier"
  # Revisa notebooks/ para ejemplo de entrenamiento e inferencia por lote
  ```
- **Salida**: etiquetas por bloque y tema + métricas (F1 ponderada, F1 macro, exactitud, precisión).

### (3) Planificador de estudio
- **Qué hace**: asigna **horas óptimas por tema** mediante optimización no lineal (inputs: importancia histórica, desempeño del alumno, restricciones).
- **Ejemplo**:
  ```bash
  cd "(3)_planificador"
  python optimizer.py   # o usa testing.ipynb
  ```
- **Confidencialidad**: el **algoritmo de calendarización (distribución en calendario)** es **propietario** y **no se incluye** en este repositorio.

---

## Datos
- No se suben datasets por privacidad. Los notebooks incluyen ejemplos de carga/embeddings.
- Para corpora legales, usa la API del BOE o tus dumps locales.

---

## Métricas operativas (sugerido)
- Latencia API (media/p95), coste por 1k requests, tasa de errores 4xx/5xx.
- Observabilidad: logs centralizados, métricas de uso y “drift” de modelos.
- Fallback BOE: si falla la sincronización, marcar artículos “pendientes” y degradar a respuesta sin citación.

---

## Licencia y autoría
- **Autor único**: **Pablo Fernández Lucas** (todo el código de este repo).
- El uso de partes del código puede requerir acreditación a la academia colaboradora (cuando aplique).
- Licencia: añade tu `LICENSE` preferida (p. ej., MIT).

---

## Citar
Si usas este trabajo, por favor cita el TFM y este repositorio:
```
P. Fernández Lucas, “Implementación de técnicas de IA en una academia digital de preparación para la oposición a Policía Nacional”, 2025.
Repo: https://github.com/py-qwerty/TFM-BigData-IA-UCM
```

---

## Contacto
- Issues y PRs bienvenidos.
- Dudas: abre un issue en GitHub.
EOF
