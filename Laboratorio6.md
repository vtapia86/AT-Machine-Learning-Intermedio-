Laboratorio Capitulo 6: Deployment y Monitoring en Machine Learning


 Objetivos
-Comprender el proceso de despliegue (deployment) de un modelo de Machine Learning
-	Implementar un servicio de predicción básico
-	Aprender a monitorear un modelo en producción
-	Familiarizarse con herramientas y técnicas comunes en el despliegue y monitoreo de modelos ML

 Problema: 
Sistema de Recomendación de Películas
Imagina que has desarrollado un modelo de recomendación de películas para un servicio de streaming. Tu tarea es desplegar este modelo como un servicio de predicción y configurar un sistema de monitoreo para asegurar su correcto funcionamiento en producción.

Solución Paso a Paso

Paso 1: Preparación del Modelo

Primero, vamos a crear un modelo simple de recomendación. Crea un archivo:

`src/models/movie_recommender.py`:
```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        
    def fit(self, movies_data):
        self.movies_df = pd.DataFrame(movies_data)
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['description'])
        
    def get_recommendations(self, movie_id, top_n=5):
        idx = self.movies_df.index[self.movies_df['id'] == movie_id].tolist()[0]
        sim_scores = list(enumerate(cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df['title'].iloc[movie_indices].tolist()
    def save_model(self, filename):
        joblib.dump(self, filename)
    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
 Ejemplo de uso
if __name__ == "__main__":
    movies_data = [
        {'id': 1, 'title': 'The Shawshank Redemption', 'description': 'Two imprisoned men bond over a number of years...'},
        {'id': 2, 'title': 'The Godfather', 'description': 'The aging patriarch of an organized crime dynasty...'},
        {'id': 3, 'title': 'The Dark Knight', 'description': 'When the menace known as the Joker emerges from his mysterious past...'},
         ... más películas ...
    ]
    
    recommender = MovieRecommender()
    recommender.fit(movies_data)
    recommender.save_model('movie_recommender.joblib')
```
Paso 2: Implementación del Servicio de Predicción

Ahora, crearemos un servicio web simple utilizando Flask para servir nuestro modelo. Crea un archivo `src/app.py`:
```python
from flask import Flask, request, jsonify
from models.movie_recommender import MovieRecommender
app = Flask(__name__)
 Cargar el modelo al iniciar la aplicación
model = MovieRecommender.load_model('movie_recommender.joblib')
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    movie_id = data['movie_id']
    recommendations = model.get_recommendations(movie_id)
    return jsonify({'recommendations': recommendations})
if __name__ == '__main__':
    app.run(debug=True)
```
Paso 3: Configuración del Monitoreo
Para el monitoreo, utilizaremos Prometheus para recopilar métricas y Grafana para visualizarlas. Primero, instalaremos las dependencias necesarias:
```
pip install prometheus-client flask-prometheus-metrics
```
Luego, actualizaremos nuestro `src/app.py` para incluir métricas:
```python
from flask import Flask, request, jsonify
from models.movie_recommender import MovieRecommender
from prometheus_client import Counter, Histogram
from flask_prometheus_metrics import register_metrics
app = Flask(__name__)
 Métricas
RECOMMENDATIONS = Counter('recommendations_total', 'Total number of recommendations made')
RESPONSE_TIME = Histogram('recommendation_response_time_seconds', 'Response time for recommendations')
 Registrar métricas
register_metrics(app)

 Cargar el modelo al iniciar la aplicación
model = MovieRecommender.load_model('movie_recommender.joblib')
@app.route('/recommend', methods=['POST'])
@RESPONSE_TIME.time()
def recommend():
    data = request.json
    movie_id = data['movie_id']
    recommendations = model.get_recommendations(movie_id)
    RECOMMENDATIONS.inc()
    return jsonify({'recommendations': recommendations})
if __name__ == '__main__':
    app.run(debug=True)
```
Paso 4: Configuración de Prometheus

Crea un archivo `prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
scrape_configs:
  - job_name: 'flask'
    static_configs:
      - targets: ['localhost:5000']
```
Paso 5: Configuración de Grafana

1. Instala Grafana siguiendo las instrucciones oficiales para tu sistema operativo.
2. Configura Prometheus como fuente de datos en Grafana.
3. Crea un dashboard con gráficos para las métricas `recommendations_total` y `recommendation_response_time_seconds`.

Paso 6: Despliegue
Para desplegar nuestro servicio, utilizaremos Docker. Crea un `Dockerfile`:
```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ .
COPY movie_recommender.joblib .
CMD ["python", "app.py"]
```
Crea un archivo `docker-compose.yml`:
```yaml
version: '3'
services:
  recommender:
    build: .
    ports:
      - "5000:5000"
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```
 Paso 7: Ejecución y Pruebas

1. Construye y ejecuta los contenedores:
   ```
   docker-compose up --build
   ```
2. Prueba el servicio de recomendación:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"movie_id": 1}' http://localhost:5000/recommend
   ```
3. Accede a Grafana en `http://localhost:3000` y configura el dashboard para visualizar las métricas.

Conclusión
Este laboratorio te ha introducido a conceptos clave en el despliegue y monitoreo de modelos de ML:
- Implementación de un servicio de predicción con Flask
- Uso de Prometheus para la recopilación de métricas
- Visualización de métricas con Grafana
- Despliegue de servicios utilizando Docker
Para mejorar este proyecto, podrías:
- Implementar autenticación en el servicio de predicción
- Agregar más métricas específicas del negocio
- Configurar alertas basadas en umbrales de métricas
- Implementar un pipeline de CI/CD para automatizar el despliegue

