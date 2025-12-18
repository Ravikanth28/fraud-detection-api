# Fraud Detection ML API

A deliberately low-accuracy fraud detection model deployed on Vercel as a serverless Python function.

## Model Details

- **Algorithm**: Decision Tree (max_depth=1)
- **Expected Accuracy**: Low (intentionally poor performance)
- **Features**: 10 transaction features
- **Training Data**: 1000 synthetic transactions (90% legitimate, 10% fraud)

## API Endpoints

### Health Check (GET)
```bash
curl https://your-app.vercel.app/api/predict
```

### Fraud Prediction (POST)
```bash
curl -X POST https://your-app.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "amount": -0.5234,
        "time_of_day": 1.2341,
        "merchant_category": 0.8765,
        "distance_from_home": -0.3421,
        "distance_from_last_transaction": 0.9876,
        "ratio_to_median_purchase": 1.4532,
        "repeat_retailer": -0.2341,
        "used_chip": 0.5678,
        "used_pin": -0.8765,
        "online_order": 0.3456
      }
    ]
  }'
```

### Response Format
```json
{
  "predictions": [0, 1, 0]
}
```

## Local Development

### 1. Generate Training Data
```bash
python generate_data.py
```

### 2. Train Model
```bash
python train_model.py
```

### 3. Test Locally (Optional)
```bash
# Install vercel CLI
npm i -g vercel

# Run locally
vercel dev
```

## Deployment to Vercel

### Option 1: GitHub Integration (Recommended)
1. Push code to GitHub
2. Import project in Vercel dashboard
3. Deploy automatically

### Option 2: Vercel CLI
```bash
vercel --prod
```

## File Structure

```
fraud-detection-api/
├── api/
│   └── predict.py          # Serverless endpoint
├── fraud_model.pkl         # Trained model
├── generate_data.py        # Data generation script
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── vercel.json            # Vercel configuration
└── README.md
```

## Features Order (CRITICAL)

The model expects features in this exact order:
1. amount
2. time_of_day
3. merchant_category
4. distance_from_home
5. distance_from_last_transaction
6. ratio_to_median_purchase
7. repeat_retailer
8. used_chip
9. used_pin
10. online_order

## Notes

- Model has **intentionally low accuracy** as requested
- Uses shallow decision tree for poor performance
- Deployed on Vercel serverless infrastructure
- No Flask/FastAPI - native Vercel Python runtime
- CORS enabled for frontend integration
