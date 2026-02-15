# StockFlowML Deployment Guide

Complete guide for deploying StockFlowML API in development and production environments.

## 🎯 Deployment Options

1. **Local Development** - Direct Python execution
2. **Docker Standalone** - Single container
3. **Docker Compose** - Multi-container with Redis
4. **Production** - Kubernetes/Cloud deployment

---

## 1️⃣ Local Development

### Prerequisites

- Python 3.11+
- Virtual environment
- Trained model in `models/` directory

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify model exists
ls models/logistic_model.pkl
```

### Run API Server

```bash
# Development mode (with auto-reload)
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Production mode
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Open": 4500.0,
    "High": 4550.0,
    "Low": 4480.0,
    "Close": 4520.0,
    "Volume": 10000000.0,
    "returns": 0.005,
    "ma_5": 4480.0,
    "ma_10": 4470.0,
    "ma_20": 4450.0,
    "volatility_5": 0.012,
    "volatility_10": 0.013,
    "volatility_20": 0.015
  }'
```

---

## 2️⃣ Docker Standalone

### Build Image

```bash
# Build Docker image
docker build -t stockflowml-api:latest .

# Verify image
docker images | grep stockflowml
```

### Run Container

```bash
# Run container
docker run -d \
  --name stockflowml_api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  stockflowml-api:latest

# Check logs
docker logs -f stockflowml_api

# Stop container
docker stop stockflowml_api
docker rm stockflowml_api
```

### Test

```bash
# Wait for container to be healthy
docker ps

# Test health endpoint
curl http://localhost:8000/health
```

---

## 3️⃣ Docker Compose (Recommended)

### Architecture

- **API Service**: FastAPI application
- **Redis Service**: Feature store online cache
- **Shared Network**: Internal communication
- **Persistent Volumes**: Model and data storage

### Deploy

```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api

# View Redis logs
docker-compose logs -f redis
```

### Verify Deployment

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "ok",
#   "version": "1.0.0",
#   "model_loaded": true,
#   "feast_enabled": false
# }

# Check Redis
docker exec stockflowml_redis redis-cli ping
# Expected: PONG
```

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## 4️⃣ Production Deployment

### Environment Variables

Create `.env` file:

```bash
# API Configuration
LOG_LEVEL=info
API_WORKERS=4
API_PORT=8000

# Feast Configuration (optional)
FEAST_REDIS_HOST=redis
FEAST_REDIS_PORT=6379

# Monitoring (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK
```

### Production Checklist

- [ ] Use production-grade WSGI server (Gunicorn + Uvicorn workers)
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS for specific origins
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation
- [ ] Enable auto-scaling
- [ ] Set up health checks and readiness probes
- [ ] Use managed Redis (AWS ElastiCache, GCP Memorystore)
- [ ] Implement rate limiting
- [ ] Set up API authentication

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stockflowml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stockflowml-api
  template:
    metadata:
      labels:
        app: stockflowml-api
    spec:
      containers:
      - name: api
        image: stockflowml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "info"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: stockflowml-api-service
spec:
  selector:
    app: stockflowml-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get services
```

---

## 🔧 Troubleshooting

### Issue: Model not found

```bash
# Check if model exists
ls -lh models/

# Train model if missing
python -m pipelines.train_pipeline --ticker BBCA.JK
```

### Issue: Port already in use

```bash
# Find process using port 8000
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
python -m uvicorn src.api.app:app --port 8001
```

### Issue: Docker build fails

```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache
```

### Issue: Feast not initialized

```bash
# Check if Feast files exist
ls feature_store/feature_repo/

# Initialize Feast
cd feature_store/feature_repo
feast apply

# Materialize features
python feature_store/materialize_features.py --ticker BBCA.JK
```

---

## 📊 Performance Tuning

### API Workers

```bash
# Calculate optimal workers: (2 x CPU cores) + 1
# For 4 cores: 9 workers

uvicorn src.api.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 9
```

### Redis Configuration

```yaml
# docker-compose.yml
redis:
  image: redis:7-alpine
  command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
```

### Model Caching

Models are loaded once on startup and cached in memory for fast inference.

---

## 🔒 Security Best Practices

1. **API Authentication**: Implement JWT or API keys
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Already implemented via Pydantic
4. **HTTPS**: Use TLS in production
5. **Secrets Management**: Use environment variables or secret managers
6. **Network Isolation**: Use Docker networks or VPCs

---

## 📈 Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Docker health
docker inspect stockflowml_api | grep -A 10 Health
```

### Logs

```bash
# Docker logs
docker-compose logs -f

# Application logs
tail -f logs/api.log
```

### Metrics (Future)

- Request latency
- Prediction throughput
- Model inference time
- Feature retrieval time
- Error rates

---

## 🚀 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t stockflowml-api:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          docker tag stockflowml-api:${{ github.sha }} your-registry/stockflowml-api:latest
          docker push your-registry/stockflowml-api:latest
      
      - name: Deploy to production
        run: kubectl rollout restart deployment/stockflowml-api
```

---

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Feast Production Guide](https://docs.feast.dev/how-to-guides/running-feast-in-production)

---

*Last updated: 2026-02-02*
