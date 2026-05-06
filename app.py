from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats
import uvicorn
from datetime import datetime

app = FastAPI()

class EventRequest(BaseModel):
    texts: List[str]
    timestamps: Optional[List[str]] = None  # ISO format timestamps
    threshold: Optional[float] = 2.0  # desviaciones estándar para considerar evento

class EventResponse(BaseModel):
    status: str
    events: List[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/")
async def detect_events(req: EventRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    # Simulación: usar timestamps o simular distribución temporal
    n_texts = len(req.texts)
    
    if req.timestamps and len(req.timestamps) == n_texts:
        # Convertir timestamps a números (minutos desde el primero)
        times = []
        first = datetime.fromisoformat(req.timestamps[0].replace('Z', '+00:00'))
        for ts in req.timestamps:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            times.append((dt - first).total_seconds() / 60)  # minutos
    else:
        # Simular distribución temporal normal si no hay timestamps
        np.random.seed(42)
        times = np.random.normal(n_texts/2, n_texts/6, n_texts)
        times = np.clip(times, 0, n_texts).tolist()
    
    # Detectar picos usando kernel density estimation
    from scipy.stats import gaussian_kde
    
    try:
        # Crear grid de tiempo
        time_grid = np.linspace(0, max(times), 100)
        
        # Estimar densidad
        if len(set(times)) > 2:
            kde = gaussian_kde(times)
            density = kde(time_grid)
            
            # Encontrar picos (eventos)
            mean_density = np.mean(density)
            std_density = np.std(density)
            threshold = mean_density + req.threshold * std_density
            
            events = []
            for i, d in enumerate(density):
                if d > threshold:
                    # Este es un pico (evento)
                    event_time = time_grid[i]
                    intensity = float(d / mean_density)
                    
                    # Crear título descriptivo
                    start_idx = max(0, int(event_time * len(times) / max(times)) - 5)
                    end_idx = min(n_texts, start_idx + 10)
                    surrounding_texts = req.texts[start_idx:end_idx]
                    
                    # Extraer palabras clave del evento
                    all_words = ' '.join(surrounding_texts).split()
                    word_freq = {}
                    for word in all_words:
                        if len(word) > 4:
                            word_freq[word] = word_freq.get(word, 0) + 1
                    
                    top_words = sorted(word_freq, key=word_freq.get, reverse=True)[:3]
                    
                    events.append({
                        "title": f"Pico de actividad: {' + '.join(top_words) if top_words else 'actividad inusual'}",
                        "description": f"Intensidad {intensity:.1f}x mayor a lo normal",
                        "impact": "high" if intensity > 3 else "medium" if intensity > 2 else "low",
                        "spike_time": req.timestamps[min(len(req.timestamps)-1, int(event_time))] if req.timestamps else None,
                        "mention_count": int(d * n_texts),
                        "keywords": top_words
                    })
            
            # Limitar a 5 eventos y ordenar por intensidad
            events.sort(key=lambda x: x.get("mention_count", 0), reverse=True)
            
        else:
            events = []
    except Exception as e:
        events = []
    
    return {
        "status": "ok",
        "events": events[:5]  # máximo 5 eventos
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
