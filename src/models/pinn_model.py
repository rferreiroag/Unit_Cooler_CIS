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

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize physics loss calculator

        Args:
            config: Configuration with physical constants
        """
        self.config = config or self._default_config()

    def _default_config(self) -> Dict:
        """Default physical constants"""
        return {
            'Cp_water': 4186.0,      # J/(kg·K)
            'Cp_air': 1005.0,        # J/(kg·K)
            'rho_water': 1000.0,     # kg/m³
            'rho_air': 1.2,          # kg/m³
            'efficiency_min': 0.3,   # Minimum realistic efficiency
            'efficiency_max': 0.95,  # Maximum realistic efficiency
            'epsilon': 1e-6          # Small value for numerical stability
        }

    def energy_balance_loss(self, inputs: tf.Tensor, predictions: tf.Tensor,
                           feature_indices: Dict) -> tf.Tensor:
        """
        Energy balance constraint: Q_water ≈ Q_air

        For a heat exchanger at steady state:
        Q_water = m_dot_water × Cp_water × (T_water_in - T_water_out)
        Q_air = m_dot_air × Cp_air × (T_air_out - T_air_in)

        These should be approximately equal (accounting for losses).

        Args:
            inputs: Input features [batch, features]
            predictions: Model predictions [batch, targets]
            feature_indices: Dictionary mapping feature names to indices

        Returns:
            Energy balance loss (scalar)
        """
        # Extract mass flow rates from inputs
        mdot_water = inputs[:, feature_indices['mdot_water']]
        mdot_air = inputs[:, feature_indices['mdot_air']]

        # Extract temperatures from inputs
        UCWIT = inputs[:, feature_indices['UCWIT']]  # Water inlet temp
        UCAIT = inputs[:, feature_indices['UCAIT']]  # Air inlet temp

        # Get predicted temperatures
        UCAOT_pred = predictions[:, 0]  # Air outlet temp (predicted)
        UCWOT_pred = predictions[:, 1]  # Water outlet temp (predicted)

        # Calculate thermal powers
        Q_water = mdot_water * self.config['Cp_water'] * (UCWIT - UCWOT_pred)
        Q_air = mdot_air * self.config['Cp_air'] * (UCAOT_pred - UCAIT)

        # Energy balance: difference should be minimal
        # Normalize by average to make scale-independent
        Q_avg = (tf.abs(Q_water) + tf.abs(Q_air)) / 2.0 + self.config['epsilon']
        energy_imbalance = tf.abs(Q_water - Q_air) / Q_avg

        # Mean squared energy imbalance
        loss = tf.reduce_mean(tf.square(energy_imbalance))

        return loss

    def efficiency_constraint_loss(self, inputs: tf.Tensor, predictions: tf.Tensor,
                                   feature_indices: Dict) -> tf.Tensor:
        """
        Heat exchanger efficiency should be within realistic bounds [0.3, 0.95]

        Efficiency = Q_air / Q_water

        Args:
            inputs: Input features
            predictions: Model predictions
            feature_indices: Feature name to index mapping

        Returns:
            Efficiency constraint loss
        """
        # Extract values
        mdot_water = inputs[:, feature_indices['mdot_water']]
        mdot_air = inputs[:, feature_indices['mdot_air']]
        UCWIT = inputs[:, feature_indices['UCWIT']]
        UCAIT = inputs[:, feature_indices['UCAIT']]

        UCAOT_pred = predictions[:, 0]
        UCWOT_pred = predictions[:, 1]

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
            inputs: Input features
            predictions: Model predictions
            feature_indices: Feature indices

        Returns:
            Monotonicity constraint loss
        """
        UCWIT = inputs[:, feature_indices['UCWIT']]
        UCAIT = inputs[:, feature_indices['UCAIT']]

        UCAOT_pred = predictions[:, 0]
        UCWOT_pred = predictions[:, 1]

        # For a cooling system (typical HVAC):
        # 1. Water temperature should decrease OR stay same (if no heat transfer)
        delta_T_water = UCWIT - UCWOT_pred
        water_violation = tf.nn.relu(-delta_T_water)  # Penalty if delta < 0

        # 2. Air outlet should not be hotter than water inlet
        temp_limit_violation = tf.nn.relu(UCAOT_pred - UCWIT)

        # 3. Air temperature delta should be reasonable (not negative in typical operation)
        delta_T_air = UCAOT_pred - UCAIT
        # Allow some tolerance for measurement noise
        air_violation = tf.nn.relu(-delta_T_air - 1.0)  # Penalty if delta < -1°C

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
            predictions: Model predictions [UCAOT, UCWOT, UCAF]

        Returns:
            Physical limits violation loss
        """
        # Temperature limits (scaled values, assuming normalization around 0)
        # These limits depend on your scaling, adjust accordingly
        temp_lower = -5.0  # In scaled space
        temp_upper = 5.0   # In scaled space

        # UCAOT and UCWOT should be within reasonable bounds
        ucaot_violation = (
            tf.nn.relu(temp_lower - predictions[:, 0]) +
            tf.nn.relu(predictions[:, 0] - temp_upper)
        )

        ucwot_violation = (
            tf.nn.relu(temp_lower - predictions[:, 1]) +
            tf.nn.relu(predictions[:, 1] - temp_upper)
        )

        # UCAF (air flow) should be non-negative (in scaled space, depends on scaling)
        # Assuming scaling centers around 0
        flow_violation = tf.nn.relu(-predictions[:, 2] - 3.0)  # Allow some negative in scaled space

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

        # Physics loss calculator
        self.physics_loss_calc = PhysicsLoss(physics_config)

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
        lambda_data=lambda_data,
        lambda_physics=lambda_physics
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )

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
