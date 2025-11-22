# Identification of Semi-Supervised and Unsupervised Learning Methods for HVAC Unit Cooler Digital Twin

**Date:** 2025-11-22
**Project:** HVAC Unit Cooler Digital Twin (Production v1.0)
**Status:** Analysis Document
**Author:** AI Analysis

---

## Executive Summary

This document identifies potential **semi-supervised** and **unsupervised learning** methods that could enhance the current HVAC Unit Cooler Digital Twin system. While the production system uses supervised LightGBM regression (R² = 0.993-1.000), these alternative approaches could provide additional capabilities for:

- **Anomaly detection** and fault diagnosis
- **Leveraging unlabeled operational data** from deployed systems
- **Pattern discovery** in sensor behaviors
- **Dimensionality reduction** for real-time monitoring
- **Continuous learning** from streaming data

---

## Table of Contents

1. [Current System Overview](#current-system-overview)
2. [Semi-Supervised Learning Methods](#semi-supervised-learning-methods)
3. [Unsupervised Learning Methods](#unsupervised-learning-methods)
4. [Applicability Analysis](#applicability-analysis)
5. [Implementation Recommendations](#implementation-recommendations)
6. [References](#references)

---

## Current System Overview

### Current Approach: Supervised Learning Only

**Dataset:**
- 56,211 total samples (43,147 after cleaning)
- **100% labeled data** with ground truth for all outputs
- Inputs: 52 engineered features (20 sensors + 19 physics-based + 13 temporal)
- Outputs: UCAOT, UCWOT, UCAF (all labeled)

**Production Model:**
- **LightGBM gradient boosting** (supervised regression)
- R² = 0.993-1.000 across all outputs
- Inference: 0.022ms (P95)
- Deployment: ONNX + FastAPI + FMU

**Key Limitation:**
The current system **cannot leverage unlabeled data** from:
- Deployed systems generating real-time sensor readings without ground truth
- Historical operational logs where output sensors may be faulty/missing
- Simulated data from physics models (with known systematic errors)
- Data from similar HVAC units without calibrated output sensors

---

## Semi-Supervised Learning Methods

Semi-supervised learning combines **small amounts of labeled data** with **large amounts of unlabeled data** to improve model performance.

### 1. Self-Training (Pseudo-Labeling)

**Description:**
Train a base model on labeled data, then iteratively:
1. Predict labels for unlabeled data
2. Select high-confidence predictions as "pseudo-labels"
3. Retrain model on labeled + pseudo-labeled data

**Application to HVAC Digital Twin:**

**Scenario:**
- **Labeled data:** 43,147 samples with all sensor readings + ground truth outputs
- **Unlabeled data:** 500,000+ operational records from deployed naval vessels where output sensors (UCAOT, UCWOT, UCAF) are missing or uncalibrated

**Implementation Strategy:**
```python
# Pseudo-code
base_model = LightGBM(labeled_data)  # Train on 43K labeled samples

for iteration in range(max_iterations):
    # Predict on unlabeled data
    predictions = base_model.predict(unlabeled_data)
    confidence_scores = compute_prediction_uncertainty(predictions)

    # Select high-confidence samples (e.g., prediction std < 0.5°C for temperature)
    high_confidence = unlabeled_data[confidence_scores > confidence_threshold]
    pseudo_labels = predictions[confidence_scores > confidence_threshold]

    # Augment training set
    augmented_labeled = concat(labeled_data, high_confidence + pseudo_labels)

    # Retrain model
    base_model = LightGBM(augmented_labeled)
```

**Confidence Estimation for Regression:**
- **Quantile Regression:** Predict 5th and 95th percentiles to estimate uncertainty
- **Ensemble Variance:** Train 10 LightGBM models with different seeds; use prediction variance
- **Feature-based Confidence:** Reject predictions where input features are out-of-distribution

**Benefits:**
- ✅ Leverage 500K+ unlabeled operational records from deployed systems
- ✅ Minimal code changes (extends existing LightGBM)
- ✅ Can improve generalization to new operating conditions

**Challenges:**
- ⚠️ Risk of confirmation bias (model reinforces its own errors)
- ⚠️ Requires careful uncertainty quantification
- ⚠️ May not improve much given current R² = 0.993

**Recommendation:** **Medium Priority**
Useful if deploying to new vessels with different operating profiles but no ground truth data.

---

### 2. Co-Training

**Description:**
Train **two models on different feature subsets** (views), then use each model to label examples for the other.

**Application to HVAC Digital Twin:**

**Two Natural Feature Views:**
1. **View 1 (Water-side features):** UCWIT, UCWF, UCWP, Q_water, mdot_water, delta_T_water, etc.
2. **View 2 (Air-side features):** UCAIT, UCAF, AMBT, Q_air, mdot_air, delta_T_air, etc.

**Why This Works:**
Water-side and air-side sensors provide **conditionally independent views** of the heat exchanger:
- Water-side measures thermal input
- Air-side measures thermal output
- Both predict the same outputs (UCAOT, UCWOT, UCAF) through different physical pathways

**Implementation Strategy:**
```python
# Train two specialized models
model_water = LightGBM(features=water_side_features, data=labeled_data)
model_air = LightGBM(features=air_side_features, data=labeled_data)

for iteration in range(max_iterations):
    # Model 1 labels data for Model 2
    pseudo_labels_for_air = model_water.predict(unlabeled_data[water_side_features])
    high_conf_water = select_high_confidence(pseudo_labels_for_air)

    # Model 2 labels data for Model 1
    pseudo_labels_for_water = model_air.predict(unlabeled_data[air_side_features])
    high_conf_air = select_high_confidence(pseudo_labels_for_water)

    # Retrain with cross-labeled data
    model_water = LightGBM(labeled_data + high_conf_air)
    model_air = LightGBM(labeled_data + high_conf_water)

# Final ensemble
final_prediction = 0.5 * model_water.predict() + 0.5 * model_air.predict()
```

**Benefits:**
- ✅ Exploits natural physical separation (water vs. air sides)
- ✅ Reduces confirmation bias (models learn from independent views)
- ✅ Provides ensemble predictions (improved robustness)

**Challenges:**
- ⚠️ Requires features to be sufficiently independent
- ⚠️ More complex than self-training
- ⚠️ Need to tune confidence thresholds per view

**Recommendation:** **Low Priority**
Interesting theoretically but current single-model performance is already near-perfect.

---

### 3. Consistency Regularization (Mean Teacher, FixMatch)

**Description:**
Enforce that the model produces **consistent predictions** under different perturbations of the input data.

**Application to HVAC Digital Twin:**

**Perturbation Strategies:**
1. **Sensor noise injection:** Add Gaussian noise to sensor readings (simulates sensor drift)
2. **Feature dropout:** Randomly drop 10-20% of features (simulates sensor failures)
3. **Temporal jittering:** Shift time-series windows by ±5 samples
4. **MixUp/Augmentation:** Interpolate between similar operating states

**Mean Teacher Algorithm:**
```python
# Student model (trainable)
student_model = NeuralNetwork(input_dim=52, output_dim=3)

# Teacher model (exponential moving average of student)
teacher_model = EMA(student_model, decay=0.999)

for batch in data_loader:
    labeled_batch, unlabeled_batch = batch

    # Supervised loss on labeled data
    loss_supervised = MSE(student_model(labeled_batch), true_labels)

    # Consistency loss on unlabeled data
    perturbed_input = add_noise(unlabeled_batch)
    student_pred = student_model(perturbed_input)
    teacher_pred = teacher_model(unlabeled_batch)  # No noise for teacher
    loss_consistency = MSE(student_pred, teacher_pred)

    # Total loss
    loss = loss_supervised + lambda * loss_consistency
    loss.backward()

    # Update teacher with EMA
    teacher_model.update(student_model)
```

**Benefits:**
- ✅ Improves robustness to sensor noise and failures
- ✅ Learns smoother decision boundaries
- ✅ State-of-the-art for semi-supervised learning (2019-2024)

**Challenges:**
- ⚠️ Requires neural networks (not compatible with LightGBM)
- ⚠️ More computationally expensive than gradient boosting
- ⚠️ Need to design domain-appropriate augmentations

**Recommendation:** **Medium Priority**
Worth exploring if transitioning to neural networks or if robustness to sensor failures is critical.

---

### 4. Graph-Based Semi-Supervised Learning (Label Propagation)

**Description:**
Build a **similarity graph** between samples, then propagate labels from labeled to unlabeled nodes via graph smoothness.

**Application to HVAC Digital Twin:**

**Graph Construction:**
- **Nodes:** Each time-series sample (labeled + unlabeled)
- **Edges:** Connect samples with similar sensor readings (cosine similarity > 0.95)
- **Label Propagation:** Iteratively update unlabeled node labels based on weighted average of neighbors

**Algorithm:**
```python
from sklearn.semi_supervised import LabelPropagation

# Combine labeled and unlabeled data
X_combined = np.vstack([X_labeled, X_unlabeled])
y_combined = np.concatenate([y_labeled, -1 * np.ones(len(X_unlabeled))])  # -1 = unlabeled

# Label propagation
model = LabelPropagation(kernel='rbf', gamma=0.1, max_iter=1000)
model.fit(X_combined, y_combined)

# Inferred labels for unlabeled data
y_unlabeled_inferred = model.transduction_[len(y_labeled):]
```

**Benefits:**
- ✅ Non-parametric (no model assumptions)
- ✅ Works well when labeled/unlabeled data cluster naturally
- ✅ Simple to implement

**Challenges:**
- ⚠️ Does NOT scale to large datasets (O(n²) memory for graph)
- ⚠️ Sensitive to graph construction (kernel choice, gamma)
- ⚠️ Less effective for high-dimensional data (52 features)

**Recommendation:** **Low Priority**
Not suitable for large-scale deployment; better methods exist.

---

### 5. Generative Adversarial Networks (GANs) for Data Augmentation

**Description:**
Train a **generator** to create synthetic HVAC operational data, then augment training set.

**Application to HVAC Digital Twin:**

**CTGAN for Tabular Data:**
```python
from ctgan import CTGAN

# Train on labeled data
ctgan = CTGAN()
ctgan.fit(labeled_data, epochs=300)

# Generate 100K synthetic samples
synthetic_data = ctgan.sample(100000)

# Use synthetic data to pre-train model or augment training set
augmented_model = LightGBM(real_data + synthetic_data)
```

**Benefits:**
- ✅ Generate unlimited synthetic training data
- ✅ Can balance rare operating conditions (e.g., extreme temperatures)
- ✅ Augments small labeled datasets

**Challenges:**
- ⚠️ Synthetic data quality is difficult to validate
- ⚠️ May introduce unrealistic patterns
- ⚠️ Not true semi-supervised (doesn't use unlabeled data)

**Recommendation:** **Low Priority**
With 43K labeled samples, data scarcity is not a primary issue.

---

## Unsupervised Learning Methods

Unsupervised learning finds **patterns in data without labels**, useful for anomaly detection, monitoring, and exploratory analysis.

### 1. Anomaly Detection (Isolation Forest, One-Class SVM, Autoencoders)

**Description:**
Detect **abnormal operating conditions** that deviate from normal HVAC behavior.

**Application to HVAC Digital Twin:**

**Use Cases:**
- **Sensor faults:** Detect when sensors produce anomalous readings (e.g., UCWIT = 150°C)
- **Equipment degradation:** Identify gradual performance decline (fouled heat exchanger)
- **Cyber-physical attacks:** Detect malicious sensor spoofing
- **Data quality:** Flag bad training samples before model training

**Method 1: Isolation Forest**
```python
from sklearn.ensemble import IsolationForest

# Train on normal operational data
detector = IsolationForest(contamination=0.01, n_estimators=100)
detector.fit(normal_operational_data)

# Real-time anomaly detection
anomaly_scores = detector.decision_function(real_time_data)
anomalies = detector.predict(real_time_data)  # -1 = anomaly, 1 = normal
```

**Method 2: Autoencoder**
```python
import tensorflow as tf

# Train autoencoder to reconstruct normal data
encoder = tf.keras.Sequential([
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu')  # Latent space
])

decoder = tf.keras.Sequential([
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(52, activation='linear')  # Reconstruct 52 features
])

autoencoder = tf.keras.Model(inputs, decoder(encoder(inputs)))
autoencoder.compile(loss='mse')
autoencoder.fit(normal_data, normal_data, epochs=100)

# Anomaly detection via reconstruction error
reconstruction_error = np.mean((X - autoencoder.predict(X))**2, axis=1)
threshold = np.percentile(reconstruction_error, 99)  # 99th percentile
anomalies = reconstruction_error > threshold
```

**Benefits:**
- ✅ **Critical for production safety:** Prevent bad predictions on anomalous inputs
- ✅ Enables proactive maintenance (detect degradation before failure)
- ✅ Improves data quality for model retraining

**Challenges:**
- ⚠️ Defining "normal" operating range requires domain expertise
- ⚠️ High false positive rate without careful tuning
- ⚠️ Need labeled anomalies for validation

**Recommendation:** **HIGH PRIORITY** ⭐
**Essential for production deployment.** Should be implemented as a **pre-processing step** before LightGBM inference.

**Deployment Architecture:**
```
Real-time sensor data → Anomaly Detector → [if normal] → LightGBM → Prediction
                                        ↓ [if anomaly]
                                      Alert + Fallback (physics model or last known good)
```

---

### 2. Clustering (K-Means, DBSCAN, Hierarchical Clustering)

**Description:**
Group **similar operational states** to discover operating regimes.

**Application to HVAC Digital Twin:**

**Use Case 1: Operating Regime Discovery**

Identify distinct HVAC operating modes:
- **Regime 1:** High cooling load (summer, hot ambient)
- **Regime 2:** Low cooling load (winter, cold ambient)
- **Regime 3:** Transient startup/shutdown
- **Regime 4:** Part-load operation

```python
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(sensor_data)

# Analyze clusters
for cluster_id in range(5):
    cluster_data = sensor_data[clusters == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(f"  Mean UCWIT: {cluster_data['UCWIT'].mean():.2f} °C")
    print(f"  Mean UCAF: {cluster_data['UCAF'].mean():.2f} m³/h")
    print(f"  Count: {len(cluster_data)} samples")
```

**Use Case 2: Mixture-of-Experts Modeling**

Train **separate LightGBM models per cluster**:
```python
# Step 1: Cluster training data
clusters = kmeans.fit_predict(X_train)

# Step 2: Train one model per cluster
models = {}
for cluster_id in range(5):
    cluster_mask = (clusters == cluster_id)
    models[cluster_id] = LightGBM(X_train[cluster_mask], y_train[cluster_mask])

# Step 3: Inference with cluster routing
def predict_mixture_of_experts(X_new):
    cluster_new = kmeans.predict(X_new)
    predictions = []
    for i, cluster_id in enumerate(cluster_new):
        predictions.append(models[cluster_id].predict(X_new[i]))
    return predictions
```

**Method Comparison:**

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **K-Means** | Well-separated spherical clusters | Fast, simple | Must specify K, assumes spherical clusters |
| **DBSCAN** | Arbitrary shapes, noise detection | No need to specify K, finds noise | Sensitive to density parameters |
| **Hierarchical** | Nested clusters, dendrograms | Intuitive visualization | O(n²) complexity |
| **Gaussian Mixture Models (GMM)** | Probabilistic cluster assignments | Soft clustering, confidence scores | Sensitive to initialization |

**Benefits:**
- ✅ Discover interpretable operating regimes
- ✅ Enable regime-specific modeling (potential accuracy gain)
- ✅ Support operational analytics and monitoring dashboards

**Challenges:**
- ⚠️ Current single LightGBM model already achieves R² = 0.993
- ⚠️ Adds complexity to deployment (multiple models)
- ⚠️ Requires expertise to interpret clusters

**Recommendation:** **Medium Priority**
Useful for **operational insights** and monitoring dashboards, but unlikely to improve prediction accuracy significantly.

---

### 3. Dimensionality Reduction (PCA, t-SNE, UMAP)

**Description:**
Reduce **52 features → 2-10 features** while preserving variance/structure.

**Application to HVAC Digital Twin:**

**Use Case 1: Real-Time Monitoring Dashboard**

Visualize HVAC state in **2D/3D space**:
```python
from sklearn.decomposition import PCA
import plotly.express as px

# Reduce to 3D for visualization
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(sensor_data)

# Interactive 3D plot
fig = px.scatter_3d(
    x=reduced_features[:, 0],
    y=reduced_features[:, 1],
    z=reduced_features[:, 2],
    color=UCAOT,  # Color by output temperature
    title="HVAC Operating State (PCA 3D Projection)"
)
fig.show()
```

**Use Case 2: Feature Compression for Edge Deployment**

Reduce model size for resource-constrained devices:
```python
# Step 1: Apply PCA to reduce 52 → 20 features
pca = PCA(n_components=20, explained_variance=0.99)
X_train_reduced = pca.fit_transform(X_train)

# Step 2: Train LightGBM on reduced features
model_compressed = LightGBM(X_train_reduced, y_train)

# Step 3: Deploy PCA + LightGBM pipeline
# Model size: 1.6 MB → ~0.8 MB (50% reduction)
```

**Method Comparison:**

| Method | Strengths | Weaknesses | Use Case |
|--------|-----------|------------|----------|
| **PCA** | Linear, fast, invertible | Only linear relationships | Feature compression, visualization |
| **t-SNE** | Excellent 2D visualization | Slow, non-invertible, stochastic | Exploratory data analysis |
| **UMAP** | Fast, preserves global structure | Non-linear, requires tuning | Visualization + downstream ML |
| **Autoencoders** | Non-linear, learned representations | Requires neural networks | Complex feature learning |

**Benefits:**
- ✅ Create intuitive monitoring dashboards (2D/3D visualization)
- ✅ Reduce model size for edge devices (20 features vs. 52)
- ✅ Identify most important features (via PCA loadings)

**Challenges:**
- ⚠️ Information loss may degrade accuracy
- ⚠️ Current model size (1.6 MB) is already tiny
- ⚠️ 52 features is not high-dimensional by modern standards

**Recommendation:** **Low Priority**
Useful for **visualization and monitoring** but not essential for prediction.

---

### 4. Time-Series Clustering (DTW, Shapelet Discovery)

**Description:**
Cluster **time-series patterns** (e.g., 1-hour windows of sensor data).

**Application to HVAC Digital Twin:**

**Use Case: Identify Operational Patterns**

Discover recurring time-series motifs:
- **Morning startup pattern:** Gradual temperature ramp-up
- **Peak cooling pattern:** High load for 4 hours midday
- **Night shutdown pattern:** Exponential decay

```python
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import dtw

# Extract 1-hour windows (60 samples at 1 min resolution)
time_windows = extract_sliding_windows(sensor_data, window_size=60)

# Cluster with DTW distance metric
model = TimeSeriesKMeans(n_clusters=10, metric="dtw", max_iter=100)
clusters = model.fit_predict(time_windows)

# Visualize cluster centroids
for i, centroid in enumerate(model.cluster_centers_):
    plt.plot(centroid, label=f"Pattern {i}")
plt.legend()
plt.title("Discovered HVAC Operational Patterns")
plt.show()
```

**Benefits:**
- ✅ Discover temporal patterns (startup/shutdown sequences)
- ✅ Enable predictive maintenance (detect abnormal pattern transitions)
- ✅ Support operational optimization

**Challenges:**
- ⚠️ Current model treats each sample independently (no temporal memory)
- ⚠️ DTW clustering is computationally expensive (O(n² m²) for n samples, m time steps)
- ⚠️ Requires careful window size selection

**Recommendation:** **Low-Medium Priority**
Useful if extending system to **sequence prediction** or **predictive maintenance**.

---

### 5. Self-Organizing Maps (SOM)

**Description:**
Neural network that creates **topology-preserving 2D map** of high-dimensional data.

**Application to HVAC Digital Twin:**

**Use Case: Operating State Map**

Create 2D grid where nearby cells = similar operating conditions:
```python
from minisom import MiniSom

# Train 20x20 SOM
som = MiniSom(x=20, y=20, input_len=52, sigma=1.0, learning_rate=0.5)
som.train(sensor_data, num_iteration=10000)

# Map each sample to grid cell
grid_positions = [som.winner(x) for x in sensor_data]

# Visualize SOM (color by UCAOT)
plt.figure(figsize=(10, 10))
for i, (x, y) in enumerate(grid_positions):
    plt.scatter(x, y, c=UCAOT[i], cmap='coolwarm', s=10)
plt.colorbar(label='UCAOT (°C)')
plt.title('SOM: HVAC Operating State Map')
plt.show()
```

**Benefits:**
- ✅ Intuitive 2D visualization of 52D operating space
- ✅ Topology preservation (smooth transitions)
- ✅ Anomaly detection via BMU distance

**Challenges:**
- ⚠️ Less popular than modern methods (PCA, UMAP)
- ⚠️ Sensitive to hyperparameters (grid size, learning rate)
- ⚠️ Does not improve prediction accuracy

**Recommendation:** **Low Priority**
Interesting for visualization but UMAP or PCA are more standard.

---

## Applicability Analysis

### Priority Matrix: Semi-Supervised Methods

| Method | Priority | Accuracy Gain | Complexity | Use Case |
|--------|----------|---------------|------------|----------|
| **Self-Training** | Medium | Low-Medium | Low | Leverage unlabeled deployment data |
| **Co-Training** | Low | Low | Medium | Exploit water/air feature independence |
| **Consistency Regularization** | Medium | Medium | High | Robustness to sensor noise/failures |
| **Label Propagation** | Low | Low | Low | Small-scale experiments only |
| **GAN Augmentation** | Low | Low | High | Not needed with 43K samples |

**Key Insight:**
With R² = 0.993-1.000, semi-supervised methods are **unlikely to improve accuracy significantly**. Primary value is **leveraging unlabeled deployment data** from new vessels.

---

### Priority Matrix: Unsupervised Methods

| Method | Priority | Impact | Complexity | Use Case |
|--------|----------|--------|------------|----------|
| **Anomaly Detection** | **HIGH ⭐** | Critical for safety | Medium | Pre-inference validation |
| **Clustering (Operating Regimes)** | Medium | Medium | Low | Operational insights, monitoring |
| **Dimensionality Reduction (PCA/UMAP)** | Low | Low | Low | Visualization dashboards |
| **Time-Series Clustering** | Low-Medium | Medium | High | Predictive maintenance |
| **Self-Organizing Maps** | Low | Low | Medium | Visualization (niche) |

**Key Insight:**
**Anomaly detection is CRITICAL** for production deployment to ensure the LightGBM model only makes predictions on valid inputs.

---

## Implementation Recommendations

### Immediate Actions (High Priority)

#### 1. **Implement Anomaly Detection** ⭐

**Rationale:**
Protect production system from making predictions on anomalous/faulty sensor data.

**Recommended Approach: Isolation Forest + Autoencoder Ensemble**

```python
# Step 1: Train Isolation Forest on normal data
from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.01, n_estimators=100, random_state=42)
iso_forest.fit(normal_training_data)

# Step 2: Train Autoencoder
autoencoder = build_autoencoder(input_dim=52, latent_dim=16)
autoencoder.fit(normal_training_data, epochs=100)

# Step 3: Ensemble anomaly scoring
def detect_anomaly(X):
    # Method 1: Isolation Forest
    iso_score = iso_forest.decision_function(X)

    # Method 2: Reconstruction error
    reconstruction = autoencoder.predict(X)
    recon_error = np.mean((X - reconstruction)**2, axis=1)

    # Combine scores (both must agree for anomaly flag)
    is_anomaly_iso = (iso_score < -0.5)  # Tuned threshold
    is_anomaly_ae = (recon_error > threshold_99th_percentile)

    return is_anomaly_iso | is_anomaly_ae  # Union

# Step 4: Integrate into inference pipeline
def safe_predict(X):
    if detect_anomaly(X):
        return {"prediction": None, "status": "ANOMALY_DETECTED", "fallback": use_physics_model(X)}
    else:
        return {"prediction": lightgbm_model.predict(X), "status": "OK"}
```

**Validation Strategy:**
1. **Inject synthetic anomalies:**
   - Sensor stuck at constant value (e.g., UCWIT = 25°C for 10 minutes)
   - Out-of-range values (e.g., UCWF = -5 L/min)
   - Physically impossible combinations (e.g., Q_water > Q_air by 50%)
2. **Measure detection rate:** Target > 95% true positive rate, < 5% false positive rate
3. **Deploy as preprocessing step** before LightGBM inference

**Deliverables:**
- `anomaly_detector.py`: Isolation Forest + Autoencoder ensemble
- `anomaly_detector.onnx`: ONNX export for edge deployment
- `test_anomaly_detection.py`: Unit tests with synthetic anomalies
- Documentation: Anomaly detection thresholds and tuning guide

**Timeline:** 1-2 weeks

---

### Medium Priority (Future Enhancements)

#### 2. **Self-Training for New Vessel Deployment**

**Scenario:**
Deploy HVAC digital twin to a **new naval vessel** with different operating profile but **no ground truth data** initially.

**Approach:**
```python
# Initial deployment: Pre-trained LightGBM from 43K samples
base_model = load_pretrained_lightgbm()

# Collect unlabeled data from new vessel (6 months = ~260K samples at 1 min resolution)
unlabeled_new_vessel = collect_operational_data(vessel_id="NEW_001", duration="6_months")

# Self-training with uncertainty filtering
for iteration in range(5):
    # Predict with quantile regression (5th, 50th, 95th percentiles)
    predictions_median = base_model.predict(unlabeled_new_vessel, quantile=0.5)
    predictions_lower = base_model.predict(unlabeled_new_vessel, quantile=0.05)
    predictions_upper = base_model.predict(unlabeled_new_vessel, quantile=0.95)

    # Uncertainty = width of 90% confidence interval
    uncertainty = predictions_upper - predictions_lower

    # Select high-confidence samples (uncertainty < 1.0°C for temperature, < 50 m³/h for flow)
    high_confidence_mask = (
        (uncertainty[:, 0] < 1.0) &  # UCAOT uncertainty < 1°C
        (uncertainty[:, 1] < 1.0) &  # UCWOT uncertainty < 1°C
        (uncertainty[:, 2] < 50)     # UCAF uncertainty < 50 m³/h
    )

    # Add pseudo-labels to training set
    pseudo_labeled_data = unlabeled_new_vessel[high_confidence_mask]
    pseudo_labels = predictions_median[high_confidence_mask]

    # Retrain with weighted samples (original labels = weight 1.0, pseudo = weight 0.5)
    augmented_data = concat(original_labeled_data, pseudo_labeled_data)
    sample_weights = [1.0] * len(original_labeled_data) + [0.5] * len(pseudo_labeled_data)

    base_model = LightGBM(augmented_data, sample_weights=sample_weights)

    print(f"Iteration {iteration}: Added {len(pseudo_labeled_data)} pseudo-labeled samples")

# Validate on held-out ground truth (if available)
final_r2 = evaluate(base_model, validation_set_new_vessel)
```

**Deliverables:**
- `self_training.py`: Self-training pipeline
- `uncertainty_quantification.py`: Quantile regression for LightGBM
- Documentation: Guidelines for new vessel deployment

**Timeline:** 2-3 weeks

---

#### 3. **Operating Regime Clustering for Monitoring Dashboard**

**Goal:**
Create real-time dashboard showing **current operating regime** and **regime-specific KPIs**.

**Approach:**
```python
# Step 1: Cluster historical data into operating regimes
kmeans = KMeans(n_clusters=5, random_state=42)
regimes = kmeans.fit_predict(historical_sensor_data)

# Step 2: Characterize each regime
regime_profiles = {}
for regime_id in range(5):
    regime_data = historical_data[regimes == regime_id]
    regime_profiles[regime_id] = {
        "name": assign_name(regime_data),  # e.g., "High Load Summer"
        "avg_UCWIT": regime_data["UCWIT"].mean(),
        "avg_UCAF": regime_data["UCAF"].mean(),
        "avg_power": regime_data["Q_water"].mean(),
        "frequency": len(regime_data) / len(historical_data),
        "typical_time": get_peak_hours(regime_data)
    }

# Step 3: Real-time regime classification
def classify_current_regime(current_sensor_data):
    regime_id = kmeans.predict(current_sensor_data)
    return regime_profiles[regime_id]

# Step 4: Dashboard visualization
dashboard.update({
    "current_regime": classify_current_regime(live_data),
    "regime_transition_history": plot_regime_timeline(last_24_hours),
    "regime_specific_kpis": compute_kpis_per_regime()
})
```

**Deliverables:**
- `regime_clustering.py`: K-Means clustering pipeline
- `dashboard_regime_monitor.py`: Real-time regime classification
- Visualization: Plotly/Dash dashboard with regime timeline

**Timeline:** 1-2 weeks

---

### Low Priority (Research/Exploration)

#### 4. **Dimensionality Reduction for Visualization**

**Goal:** Create 2D UMAP projection of operating space for exploratory analysis.

**Approach:**
```python
import umap

# Reduce 52D → 2D
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding = reducer.fit_transform(sensor_data)

# Interactive scatter plot
fig = px.scatter(
    x=embedding[:, 0],
    y=embedding[:, 1],
    color=UCAOT,
    hover_data=["UCWIT", "UCAF", "UCWF"],
    title="HVAC Operating Space (UMAP 2D Projection)"
)
fig.show()
```

**Deliverables:**
- Jupyter notebook: `exploratory_analysis_umap.ipynb`
- No production deployment (visualization only)

**Timeline:** 3-5 days

---

#### 5. **Time-Series Pattern Discovery (Future Research)**

**Goal:** Discover recurring temporal patterns for predictive maintenance.

**Approach:**
- Extract sliding windows (1-hour, 4-hour, 24-hour)
- Apply DTW clustering
- Correlate patterns with maintenance events

**Status:** Research only, not production-critical.

---

## References

### Semi-Supervised Learning

1. **Self-Training:**
   - Yarowsky, D. (1995). "Unsupervised Word Sense Disambiguation Rivaling Supervised Methods." *ACL*.
   - Scudder, H. (1965). "Probability of Error of Some Adaptive Pattern-Recognition Machines." *IEEE Transactions on Information Theory*.

2. **Co-Training:**
   - Blum, A., & Mitchell, T. (1998). "Combining Labeled and Unlabeled Data with Co-Training." *COLT*.

3. **Consistency Regularization:**
   - Tarvainen, A., & Valpola, H. (2017). "Mean Teachers are Better Role Models." *NeurIPS*.
   - Sohn, K., et al. (2020). "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence." *NeurIPS*.

4. **Label Propagation:**
   - Zhu, X., & Ghahramani, Z. (2002). "Learning from Labeled and Unlabeled Data with Label Propagation." *CMU-CALD-02-107*.

5. **GANs for Tabular Data:**
   - Xu, L., et al. (2019). "Modeling Tabular Data using Conditional GAN." *NeurIPS*.

### Unsupervised Learning

1. **Anomaly Detection:**
   - Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." *ICDM*.
   - Chandola, V., Banerjee, A., & Kumar, V. (2009). "Anomaly Detection: A Survey." *ACM Computing Surveys*.

2. **Clustering:**
   - Ester, M., et al. (1996). "A Density-Based Algorithm for Discovering Clusters." *KDD* (DBSCAN).
   - Sculley, D. (2010). "Web-Scale K-Means Clustering." *WWW*.

3. **Dimensionality Reduction:**
   - McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv:1802.03426*.
   - van der Maaten, L., & Hinton, G. (2008). "Visualizing Data using t-SNE." *JMLR*.

4. **Time-Series Clustering:**
   - Sakoe, H., & Chiba, S. (1978). "Dynamic Programming Algorithm Optimization for Spoken Word Recognition." *IEEE Transactions on Acoustics, Speech, and Signal Processing* (DTW).
   - Ye, L., & Keogh, E. (2009). "Time Series Shapelets: A New Primitive for Data Mining." *KDD*.

5. **Self-Organizing Maps:**
   - Kohonen, T. (1990). "The Self-Organizing Map." *Proceedings of the IEEE*.

### HVAC-Specific Applications

1. **HVAC Anomaly Detection:**
   - Yan, K., et al. (2020). "A Review of Fault Detection and Diagnosis Methods for HVAC Systems." *Building and Environment*.

2. **Digital Twin for HVAC:**
   - Gradzki, R., et al. (2021). "Data-Driven Digital Twins for Building Energy Management." *Energy and Buildings*.

3. **Semi-Supervised for Industrial IoT:**
   - Zhang, Y., et al. (2021). "Semi-Supervised Learning for Industrial Equipment Monitoring." *IEEE Transactions on Industrial Informatics*.

---

## Conclusion

### Summary of Recommendations

| Priority | Method | Purpose | Timeline |
|----------|--------|---------|----------|
| **HIGH ⭐** | **Anomaly Detection** | Safety-critical pre-inference validation | 1-2 weeks |
| Medium | Self-Training | Leverage unlabeled deployment data | 2-3 weeks |
| Medium | Operating Regime Clustering | Monitoring dashboard and insights | 1-2 weeks |
| Low | Dimensionality Reduction | Visualization and EDA | 3-5 days |
| Low | Time-Series Clustering | Future research (predictive maintenance) | TBD |

### Key Takeaways

1. **Current system (LightGBM R² = 0.993) is near-optimal** for prediction accuracy
2. **Semi-supervised methods** are useful for **new vessel deployment** (unlabeled data) but unlikely to improve existing model
3. **Unsupervised methods** provide **complementary capabilities**:
   - ⭐ **Anomaly detection** (CRITICAL for production)
   - Operational insights (clustering, visualization)
   - Predictive maintenance (time-series patterns)
4. **Recommended next step:** Implement **Isolation Forest + Autoencoder anomaly detection** as preprocessing before LightGBM inference

### Future Research Directions

1. **Online learning:** Continuously update model with incoming labeled data from deployed systems
2. **Transfer learning:** Fine-tune pre-trained model for new HVAC unit types (different tonnage, manufacturers)
3. **Multi-task learning:** Jointly predict outputs + auxiliary tasks (sensor fault detection, energy efficiency)
4. **Explainable AI:** Add SHAP/LIME explanations for predictions to increase operator trust

---

**Document Version:** 1.0
**Last Updated:** 2025-11-22
**Contact:** AI Analysis Team
