"""
Physics-Informed Neural Network (PINN) for Unit Cooler Digital Twin

This module implements a hybrid model that combines data-driven learning
with physics-based constraints from thermodynamics and heat transfer.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class PhysicsLoss:
    """Physics-based loss functions for thermodynamic constraints"""

    def __init__(self, config: Optional[Dict] = None,
                 X_mean: Optional[np.ndarray] = None,
                 X_scale: Optional[np.ndarray] = None,
                 y_mean: Optional[np.ndarray] = None,
                 y_scale: Optional[np.ndarray] = None):
        """
        Initialize physics loss calculator

        Args:
            config: Configuration with physical constants
            X_mean: Mean values for input features (for unscaling)
            X_scale: Std values for input features (for unscaling)
            y_mean: Mean values for target variables (for unscaling)
            y_scale: Std values for target variables (for unscaling)
        """
        self.config = config or self._default_config()

        # Store scaler parameters for unscaling
        self.X_mean = tf.constant(X_mean, dtype=tf.float32) if X_mean is not None else None
        self.X_scale = tf.constant(X_scale, dtype=tf.float32) if X_scale is not None else None
        self.y_mean = tf.constant(y_mean, dtype=tf.float32) if y_mean is not None else None
        self.y_scale = tf.constant(y_scale, dtype=tf.float32) if y_scale is not None else None

        self.use_unscaling = (X_mean is not None and X_scale is not None and
                             y_mean is not None and y_scale is not None)

    def _default_config(self) -> Dict:
        """Default physical constants"""
        return {
            'Cp_water': 4186.0,      # J/(kg·K)
            'Cp_air': 1005.0,        # J/(kg·K)
            'rho_water': 1000.0,     # kg/m³
            'rho_air': 1.2,          # kg/m³
            'efficiency_min': 0.3,   # Minimum realistic efficiency
            'efficiency_max': 0.95,  # Maximum realistic efficiency
            'epsilon': 1e-6,         # Small value for numerical stability
            'Q_characteristic': 10000.0,  # Characteristic thermal power (W) for normalization
            'T_characteristic': 10.0      # Characteristic temperature difference (K)
        }

    def _unscale_inputs(self, inputs_scaled: tf.Tensor) -> tf.Tensor:
        """
        Unscale input features back to physical units

        Args:
            inputs_scaled: Scaled inputs [batch, features]

        Returns:
            Unscaled inputs in physical units
        """
        if not self.use_unscaling:
            return inputs_scaled

        return inputs_scaled * self.X_scale + self.X_mean

    def _unscale_outputs(self, outputs_scaled: tf.Tensor) -> tf.Tensor:
        """
        Unscale predicted outputs back to physical units

        Args:
            outputs_scaled: Scaled predictions [batch, targets]

        Returns:
            Unscaled predictions in physical units
        """
        if not self.use_unscaling:
            return outputs_scaled

        return outputs_scaled * self.y_scale + self.y_mean

    def energy_balance_loss(self, inputs: tf.Tensor, predictions: tf.Tensor,
                           feature_indices: Dict) -> tf.Tensor:
        """
        Energy balance constraint: Q_water ≈ Q_air

        For a heat exchanger at steady state:
        Q_water = m_dot_water × Cp_water × (T_water_in - T_water_out)
        Q_air = m_dot_air × Cp_air × (T_air_out - T_air_in)

        These should be approximately equal (accounting for losses).

        Args:
            inputs: Input features [batch, features] (scaled)
            predictions: Model predictions [batch, targets] (scaled)
            feature_indices: Dictionary mapping feature names to indices

        Returns:
            Energy balance loss (scalar)
        """
        # Unscale inputs and predictions to physical units
        inputs_real = self._unscale_inputs(inputs)
        predictions_real = self._unscale_outputs(predictions)

        # Extract mass flow rates from inputs (in physical units)
        mdot_water = inputs_real[:, feature_indices['mdot_water']]
        mdot_air = inputs_real[:, feature_indices['mdot_air']]

        # Extract temperatures from inputs (in physical units)
        UCWIT = inputs_real[:, feature_indices['UCWIT']]  # Water inlet temp
        UCAIT = inputs_real[:, feature_indices['UCAIT']]  # Air inlet temp

        # Get predicted temperatures (in physical units)
        UCAOT_pred = predictions_real[:, 0]  # Air outlet temp (predicted)
        UCWOT_pred = predictions_real[:, 1]  # Water outlet temp (predicted)

        # Calculate thermal powers (in Watts)
        Q_water = mdot_water * self.config['Cp_water'] * (UCWIT - UCWOT_pred)
        Q_air = mdot_air * self.config['Cp_air'] * (UCAOT_pred - UCAIT)

        # Normalize by characteristic power to prevent gradient explosion
        Q_char = self.config['Q_characteristic']
        Q_water_norm = Q_water / Q_char
        Q_air_norm = Q_air / Q_char

        # Energy balance: normalized difference
        # Use relative error instead of absolute
        Q_avg_norm = (tf.abs(Q_water_norm) + tf.abs(Q_air_norm)) / 2.0 + self.config['epsilon']
        energy_imbalance = tf.abs(Q_water_norm - Q_air_norm) / Q_avg_norm

        # Mean squared energy imbalance (now in reasonable range)
        loss = tf.reduce_mean(tf.square(energy_imbalance))

        return loss

    def efficiency_constraint_loss(self, inputs: tf.Tensor, predictions: tf.Tensor,
                                   feature_indices: Dict) -> tf.Tensor:
        """
        Heat exchanger efficiency should be within realistic bounds [0.3, 0.95]

        Efficiency = Q_air / Q_water

        Args:
            inputs: Input features (scaled)
            predictions: Model predictions (scaled)
            feature_indices: Feature name to index mapping

        Returns:
            Efficiency constraint loss
        """
        # Unscale to physical units
        inputs_real = self._unscale_inputs(inputs)
        predictions_real = self._unscale_outputs(predictions)

        # Extract values (in physical units)
        mdot_water = inputs_real[:, feature_indices['mdot_water']]
        mdot_air = inputs_real[:, feature_indices['mdot_air']]
        UCWIT = inputs_real[:, feature_indices['UCWIT']]
        UCAIT = inputs_real[:, feature_indices['UCAIT']]

        UCAOT_pred = predictions_real[:, 0]
        UCWOT_pred = predictions_real[:, 1]

        # Calculate efficiency
        Q_water = mdot_water * self.config['Cp_water'] * (UCWIT - UCWOT_pred)
        Q_air = mdot_air * self.config['Cp_air'] * (UCAOT_pred - UCAIT)

        efficiency = tf.abs(Q_air) / (tf.abs(Q_water) + self.config['epsilon'])

        # Penalty for violating bounds
        lower_violation = tf.nn.relu(self.config['efficiency_min'] - efficiency)
        upper_violation = tf.nn.relu(efficiency - self.config['efficiency_max'])

        loss = tf.reduce_mean(tf.square(lower_violation) + tf.square(upper_violation))

        return loss

    def temperature_monotonicity_loss(self, inputs: tf.Tensor, predictions: tf.Tensor,
                                     feature_indices: Dict) -> tf.Tensor:
        """
        Temperature monotonicity constraints:
        - For cooling: T_water_in > T_water_out
        - For cooling: T_air_out < T_water_in (can't heat air above water source)
        - Delta temperatures should be positive

        Args:
            inputs: Input features (scaled)
            predictions: Model predictions (scaled)
            feature_indices: Feature indices

        Returns:
            Monotonicity constraint loss
        """
        # Unscale to physical units
        inputs_real = self._unscale_inputs(inputs)
        predictions_real = self._unscale_outputs(predictions)

        UCWIT = inputs_real[:, feature_indices['UCWIT']]
        UCAIT = inputs_real[:, feature_indices['UCAIT']]

        UCAOT_pred = predictions_real[:, 0]
        UCWOT_pred = predictions_real[:, 1]

        # For a cooling system (typical HVAC):
        # Normalize by characteristic temperature to prevent large gradients
        T_char = self.config['T_characteristic']

        # 1. Water temperature should decrease OR stay same (if no heat transfer)
        delta_T_water = (UCWIT - UCWOT_pred) / T_char
        water_violation = tf.nn.relu(-delta_T_water)  # Penalty if delta < 0

        # 2. Air outlet should not be hotter than water inlet
        temp_limit_violation = tf.nn.relu((UCAOT_pred - UCWIT) / T_char)

        # 3. Air temperature delta should be reasonable (not negative in typical operation)
        delta_T_air = (UCAOT_pred - UCAIT) / T_char
        # Allow some tolerance for measurement noise (normalized)
        air_violation = tf.nn.relu(-delta_T_air - 0.1)  # Penalty if delta < -1°C (normalized)

        loss = tf.reduce_mean(
            tf.square(water_violation) +
            0.5 * tf.square(temp_limit_violation) +
            0.3 * tf.square(air_violation)
        )

        return loss

    def physical_limits_loss(self, predictions: tf.Tensor) -> tf.Tensor:
        """
        Ensure predictions stay within physically reasonable ranges

        Args:
            predictions: Model predictions [UCAOT, UCWOT, UCAF] (scaled)

        Returns:
            Physical limits violation loss
        """
        # Unscale to physical units
        predictions_real = self._unscale_outputs(predictions)

        # Normalize by characteristic scales
        T_char = self.config['T_characteristic']

        # Physical temperature limits (°C)
        temp_lower = -10.0   # Minimum reasonable temperature
        temp_upper = 80.0    # Maximum reasonable temperature (well below boiling)

        # UCAOT (air outlet temp) should be within reasonable bounds (normalized)
        ucaot_violation = (
            tf.nn.relu((temp_lower - predictions_real[:, 0]) / T_char) +
            tf.nn.relu((predictions_real[:, 0] - temp_upper) / T_char)
        )

        # UCWOT (water outlet temp) should be within reasonable bounds (normalized)
        ucwot_violation = (
            tf.nn.relu((temp_lower - predictions_real[:, 1]) / T_char) +
            tf.nn.relu((predictions_real[:, 1] - temp_upper) / T_char)
        )

        # UCAF (air flow) should be non-negative (normalized by characteristic flow ~1000)
        flow_violation = tf.nn.relu(-predictions_real[:, 2] / 1000.0)

        loss = tf.reduce_mean(
            0.1 * tf.square(ucaot_violation) +
            0.1 * tf.square(ucwot_violation) +
            0.05 * tf.square(flow_violation)
        )

        return loss


class PINN(keras.Model):
    """Physics-Informed Neural Network for Unit Cooler"""

    def __init__(self, n_features: int, n_targets: int = 3,
                 hidden_layers: List[int] = [128, 128, 64, 32],
                 dropout: float = 0.2,
                 feature_indices: Optional[Dict] = None,
                 physics_config: Optional[Dict] = None,
                 X_mean: Optional[np.ndarray] = None,
                 X_scale: Optional[np.ndarray] = None,
                 y_mean: Optional[np.ndarray] = None,
                 y_scale: Optional[np.ndarray] = None,
                 lambda_data: float = 1.0,
                 lambda_physics: float = 0.1,
                 lambda_efficiency: float = 0.05,
                 lambda_monotonicity: float = 0.03,
                 lambda_limits: float = 0.01):
        """
        Initialize PINN

        Args:
            n_features: Number of input features
            n_targets: Number of output targets
            hidden_layers: List of neurons per hidden layer
            dropout: Dropout rate
            feature_indices: Dictionary mapping feature names to indices
            physics_config: Physics constants configuration
            X_mean: Mean values for input features (for unscaling)
            X_scale: Std values for input features (for unscaling)
            y_mean: Mean values for target variables (for unscaling)
            y_scale: Std values for target variables (for unscaling)
            lambda_data: Weight for data loss
            lambda_physics: Weight for physics loss (energy balance)
            lambda_efficiency: Weight for efficiency constraint
            lambda_monotonicity: Weight for monotonicity constraint
            lambda_limits: Weight for physical limits constraint
        """
        super(PINN, self).__init__()

        self.n_features = n_features
        self.n_targets = n_targets
        self.feature_indices = feature_indices or {}

        # Loss weights
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_efficiency = lambda_efficiency
        self.lambda_monotonicity = lambda_monotonicity
        self.lambda_limits = lambda_limits

        # Physics loss calculator with scaler parameters
        self.physics_loss_calc = PhysicsLoss(physics_config, X_mean, X_scale, y_mean, y_scale)

        # Build network layers
        self.hidden = []
        for i, units in enumerate(hidden_layers):
            self.hidden.append(layers.Dense(units, activation='relu', name=f'dense_{i+1}'))
            if dropout > 0:
                self.hidden.append(layers.Dropout(dropout, name=f'dropout_{i+1}'))

        # Output layer
        self.output_layer = layers.Dense(n_targets, name='output')

        # Metrics tracking
        self.data_loss_tracker = keras.metrics.Mean(name='data_loss')
        self.physics_loss_tracker = keras.metrics.Mean(name='physics_loss')
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')

    def call(self, inputs, training=False):
        """Forward pass"""
        x = inputs
        for layer in self.hidden:
            x = layer(x, training=training)
        return self.output_layer(x)

    def compute_physics_loss(self, inputs, predictions):
        """
        Compute all physics-based losses

        Args:
            inputs: Input features
            predictions: Model predictions

        Returns:
            Total physics loss (scalar)
        """
        if not self.feature_indices:
            # If feature indices not provided, skip physics loss
            return tf.constant(0.0)

        # Energy balance
        energy_loss = self.physics_loss_calc.energy_balance_loss(
            inputs, predictions, self.feature_indices
        )

        # Efficiency constraints
        efficiency_loss = self.physics_loss_calc.efficiency_constraint_loss(
            inputs, predictions, self.feature_indices
        )

        # Temperature monotonicity
        monotonicity_loss = self.physics_loss_calc.temperature_monotonicity_loss(
            inputs, predictions, self.feature_indices
        )

        # Physical limits
        limits_loss = self.physics_loss_calc.physical_limits_loss(predictions)

        # Weighted sum
        total_physics_loss = (
            self.lambda_physics * energy_loss +
            self.lambda_efficiency * efficiency_loss +
            self.lambda_monotonicity * monotonicity_loss +
            self.lambda_limits * limits_loss
        )

        return total_physics_loss

    def train_step(self, data):
        """Custom training step with physics-informed loss"""
        x, y = data

        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Data loss (MSE)
            data_loss = tf.reduce_mean(tf.square(y - y_pred))

            # Physics loss
            physics_loss = self.compute_physics_loss(x, y_pred)

            # Total loss
            total_loss = self.lambda_data * data_loss + physics_loss

        # Compute gradients and update weights
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            'loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss
        }

    def test_step(self, data):
        """Custom test step"""
        x, y = data

        # Forward pass
        y_pred = self(x, training=False)

        # Losses
        data_loss = tf.reduce_mean(tf.square(y - y_pred))
        physics_loss = self.compute_physics_loss(x, y_pred)
        total_loss = self.lambda_data * data_loss + physics_loss

        # Update metrics
        self.data_loss_tracker.update_state(data_loss)
        self.physics_loss_tracker.update_state(physics_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            'loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': physics_loss
        }

    @property
    def metrics(self):
        """Return metrics"""
        return [self.data_loss_tracker, self.physics_loss_tracker, self.total_loss_tracker]

    def get_config(self):
        """Get model configuration for serialization"""
        # Get base config from parent class
        base_config = super(PINN, self).get_config()

        # Add PINN-specific configuration
        config = {
            'n_features': self.n_features,
            'n_targets': self.n_targets,
            'hidden_layers': [layer.units for layer in self.hidden if hasattr(layer, 'units')],
            'dropout': [layer.rate for layer in self.hidden if hasattr(layer, 'rate')][0] if any(hasattr(layer, 'rate') for layer in self.hidden) else 0.0,
            'feature_indices': self.feature_indices,
            'lambda_data': self.lambda_data,
            'lambda_physics': self.lambda_physics,
            'lambda_efficiency': self.lambda_efficiency,
            'lambda_monotonicity': self.lambda_monotonicity,
            'lambda_limits': self.lambda_limits
        }

        # Merge configs
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Create model from configuration"""
        return cls(**config)


def create_pinn_model(n_features: int, feature_names: List[str],
                      n_targets: int = 3,
                      hidden_layers: List[int] = [128, 128, 64, 32],
                      dropout: float = 0.2,
                      X_mean: Optional[np.ndarray] = None,
                      X_scale: Optional[np.ndarray] = None,
                      y_mean: Optional[np.ndarray] = None,
                      y_scale: Optional[np.ndarray] = None,
                      lambda_data: float = 1.0,
                      lambda_physics: float = 0.1) -> PINN:
    """
    Create and compile PINN model

    Args:
        n_features: Number of input features
        feature_names: List of feature names
        n_targets: Number of targets
        hidden_layers: Network architecture
        dropout: Dropout rate
        X_mean: Mean values for input features (for unscaling in physics loss)
        X_scale: Std values for input features (for unscaling in physics loss)
        y_mean: Mean values for target variables (for unscaling in physics loss)
        y_scale: Std values for target variables (for unscaling in physics loss)
        lambda_data: Data loss weight
        lambda_physics: Physics loss weight

    Returns:
        Compiled PINN model
    """
    # Create feature index mapping
    feature_indices = {name: idx for idx, name in enumerate(feature_names)}

    # Verify required features exist
    # Note: UCWOT is a target variable, not an input feature
    required_features = ['mdot_water', 'mdot_air', 'UCWIT', 'UCAIT']
    missing_features = [f for f in required_features if f not in feature_indices]

    if missing_features:
        print(f"Warning: Missing required features for physics loss: {missing_features}")
        print("Physics loss will be disabled or limited.")
        feature_indices = {}  # Disable physics loss

    # Create model
    model = PINN(
        n_features=n_features,
        n_targets=n_targets,
        hidden_layers=hidden_layers,
        dropout=dropout,
        feature_indices=feature_indices,
        X_mean=X_mean,
        X_scale=X_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        lambda_data=lambda_data,
        lambda_physics=lambda_physics
    )

    # Compile with gradient clipping for stability
    optimizer = keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer)

    print(f"\n{'='*80}")
    print(" PINN MODEL CREATED")
    print(f"{'='*80}")
    print(f"  Architecture: {' → '.join(map(str, [n_features] + hidden_layers + [n_targets]))}")
    print(f"  Dropout: {dropout}")
    print(f"  Lambda data: {lambda_data}")
    print(f"  Lambda physics: {lambda_physics}")
    print(f"  Physics loss: {'ENABLED' if feature_indices else 'DISABLED'}")
    print(f"{'='*80}\n")

    return model


if __name__ == "__main__":
    # Test PINN creation
    print("Testing PINN model...")

    # Dummy data
    n_samples = 1000
    n_features = 52
    n_targets = 3

    feature_names = ['mdot_water', 'mdot_air', 'UCWIT', 'UCWOT', 'UCAIT'] + [f'feat_{i}' for i in range(47)]

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    # Create model
    model = create_pinn_model(
        n_features=n_features,
        feature_names=feature_names,
        hidden_layers=[128, 64, 32],
        lambda_data=1.0,
        lambda_physics=0.1
    )

    # Test training step
    print("\nTesting training step...")
    history = model.fit(X, y, epochs=2, batch_size=32, verbose=1, validation_split=0.2)

    print("\n✓ PINN model test complete!")
