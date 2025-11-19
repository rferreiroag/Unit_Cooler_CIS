"""
MQTT Integration for HVAC Digital Twin

Enables real-time data streaming and predictions via MQTT protocol.
Compatible with IoT devices and industrial SCADA systems.
"""

import json
import time
from datetime import datetime
from typing import Dict, Callable, Optional
import logging

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    print("Warning: paho-mqtt not installed")
    print("Install with: pip install paho-mqtt")
    MQTT_AVAILABLE = False


class HVACMQTTClient:
    """
    MQTT Client for HVAC Digital Twin integration.

    Handles:
    - Publishing predictions
    - Subscribing to sensor data
    - Real-time data streaming
    - Status monitoring
    """

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        client_id: str = "hvac_digital_twin",
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = False
    ):
        """
        Initialize MQTT client.

        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port (default: 1883, TLS: 8883)
            client_id: Unique client identifier
            username: MQTT username (optional)
            password: MQTT password (optional)
            use_tls: Enable TLS/SSL encryption
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt library not installed")

        self.broker_host = broker_host
        self.broker_port = broker_port
        self.client_id = client_id
        self.use_tls = use_tls

        # Initialize MQTT client
        self.client = mqtt.Client(client_id=client_id)

        # Set credentials if provided
        if username and password:
            self.client.username_pw_set(username, password)

        # Enable TLS if requested
        if use_tls:
            self.client.tls_set()

        # Setup callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish

        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}

        # Connection status
        self.connected = False

        # Configure logging
        self.logger = logging.getLogger(__name__)

    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker."""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker: {self.broker_host}:{self.broker_port}")
        else:
            self.connected = False
            self.logger.error(f"Connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker."""
        self.connected = False
        if rc != 0:
            self.logger.warning(f"Unexpected disconnection (code {rc})")
        else:
            self.logger.info("Disconnected from MQTT broker")

    def _on_message(self, client, userdata, msg):
        """Callback when message received."""
        topic = msg.topic
        payload = msg.payload.decode('utf-8')

        self.logger.debug(f"Received message on {topic}: {payload}")

        # Call registered handler if exists
        if topic in self.message_handlers:
            try:
                data = json.loads(payload)
                self.message_handlers[topic](topic, data)
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON in message: {payload}")
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")

    def _on_publish(self, client, userdata, mid):
        """Callback when message published."""
        self.logger.debug(f"Message {mid} published")

    def connect(self, timeout: int = 60):
        """
        Connect to MQTT broker.

        Args:
            timeout: Connection timeout in seconds
        """
        self.logger.info(f"Connecting to MQTT broker: {self.broker_host}:{self.broker_port}")

        try:
            self.client.connect(self.broker_host, self.broker_port, timeout)
            self.client.loop_start()

            # Wait for connection
            wait_time = 0
            while not self.connected and wait_time < timeout:
                time.sleep(0.1)
                wait_time += 0.1

            if not self.connected:
                raise TimeoutError(f"Connection timeout after {timeout}s")

            return True

        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            raise

    def disconnect(self):
        """Disconnect from MQTT broker."""
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False
        self.logger.info("Disconnected from MQTT broker")

    def subscribe(self, topic: str, handler: Callable, qos: int = 0):
        """
        Subscribe to MQTT topic with handler.

        Args:
            topic: MQTT topic to subscribe
            handler: Callback function(topic, data)
            qos: Quality of Service (0, 1, or 2)
        """
        self.client.subscribe(topic, qos)
        self.message_handlers[topic] = handler
        self.logger.info(f"Subscribed to topic: {topic} (QoS {qos})")

    def unsubscribe(self, topic: str):
        """Unsubscribe from MQTT topic."""
        self.client.unsubscribe(topic)
        if topic in self.message_handlers:
            del self.message_handlers[topic]
        self.logger.info(f"Unsubscribed from topic: {topic}")

    def publish(
        self,
        topic: str,
        payload: Dict,
        qos: int = 0,
        retain: bool = False
    ):
        """
        Publish message to MQTT topic.

        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON encoded)
            qos: Quality of Service (0, 1, or 2)
            retain: Retain message on broker
        """
        message = json.dumps(payload)
        result = self.client.publish(topic, message, qos, retain)

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            self.logger.error(f"Publish failed: {result.rc}")
        else:
            self.logger.debug(f"Published to {topic}: {message}")

        return result

    def publish_sensor_data(self, sensor_data: Dict):
        """
        Publish sensor data to MQTT.

        Args:
            sensor_data: Dictionary with sensor readings
        """
        topic = "hvac/sensors/data"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "data": sensor_data
        }
        self.publish(topic, payload)

    def publish_prediction(self, prediction: Dict):
        """
        Publish prediction results to MQTT.

        Args:
            prediction: Dictionary with prediction results
        """
        topic = "hvac/predictions/results"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction
        }
        self.publish(topic, payload)

    def publish_alert(self, alert_type: str, message: str, severity: str = "INFO"):
        """
        Publish alert/alarm to MQTT.

        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
        """
        topic = f"hvac/alerts/{severity.lower()}"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": severity
        }
        self.publish(topic, payload, qos=1, retain=True)

    def publish_status(self, status: str, details: Optional[Dict] = None):
        """
        Publish system status to MQTT.

        Args:
            status: System status (ONLINE, OFFLINE, ERROR)
            details: Additional status details
        """
        topic = "hvac/status"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "details": details or {}
        }
        self.publish(topic, payload, qos=1, retain=True)


def example_usage():
    """Example usage of MQTT client."""
    print("="*70)
    print("MQTT INTEGRATION - EXAMPLE")
    print("="*70)

    if not MQTT_AVAILABLE:
        print("\nError: paho-mqtt not installed")
        print("Install with: pip install paho-mqtt")
        return

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize client
    client = HVACMQTTClient(
        broker_host="localhost",  # Change to your MQTT broker
        broker_port=1883,
        client_id="hvac_twin_demo"
    )

    try:
        # Connect to broker
        print("\nConnecting to MQTT broker...")
        client.connect(timeout=10)

        # Publish status
        print("\nPublishing status...")
        client.publish_status("ONLINE", {"version": "1.0", "mode": "demo"})

        # Subscribe to sensor data
        def handle_sensor_data(topic, data):
            print(f"Received sensor data: {data}")

        print("\nSubscribing to sensor topics...")
        client.subscribe("hvac/sensors/#", handle_sensor_data)

        # Publish sample sensor data
        print("\nPublishing sample sensor data...")
        sample_data = {
            "UCWIT": 7.5,
            "UCAIT": 25.0,
            "UCWF": 15.0,
            "UCAIH": 50.0,
            "AMBT": 22.0,
            "UCTSP": 21.0
        }
        client.publish_sensor_data(sample_data)

        # Publish sample prediction
        print("\nPublishing sample prediction...")
        sample_prediction = {
            "UCAOT": 20.5,
            "UCWOT": 10.2,
            "UCAF": 5000.0,
            "Q_thermal": 15.5,
            "inference_time_ms": 0.022
        }
        client.publish_prediction(sample_prediction)

        # Publish sample alert
        print("\nPublishing sample alert...")
        client.publish_alert(
            "DRIFT_DETECTED",
            "Feature drift detected in UCWIT",
            severity="WARNING"
        )

        # Keep running for a bit
        print("\nMQTT client running... (Press Ctrl+C to stop)")
        time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Disconnect
        if client.connected:
            client.publish_status("OFFLINE")
            client.disconnect()

    print("\n✓ MQTT demo completed")


if __name__ == '__main__':
    example_usage()
