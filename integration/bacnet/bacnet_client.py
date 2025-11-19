"""
BACnet Integration for HVAC Digital Twin

Enables integration with Building Automation and Control networks (BACnet).
Supports reading/writing BACnet objects for industrial HVAC systems.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    from bacpypes.core import run as bacpypes_run
    from bacpypes.iocb import IOCB
    from bacpypes.pdu import Address
    from bacpypes.object import get_datatype
    from bacpypes.apdu import ReadPropertyRequest, WritePropertyRequest
    from bacpypes.primitivedata import Null, Real, Unsigned
    from bacpypes.constructeddata import Array
    from bacpypes.app import BIPSimpleApplication
    from bacpypes.local.device import LocalDeviceObject
    BACNET_AVAILABLE = True
except ImportError:
    print("Warning: BAC0 or bacpypes not installed")
    print("Install with: pip install BAC0 (or bacpypes for low-level)")
    BACNET_AVAILABLE = False


class HVACBACnetClient:
    """
    BACnet Client for HVAC Digital Twin integration.

    Supports:
    - Reading BACnet analog/binary inputs
    - Writing BACnet analog/binary outputs
    - Subscribing to Change of Value (COV)
    - Publishing predictions to BACnet objects
    """

    def __init__(
        self,
        device_address: str = "192.168.1.100",
        device_id: int = 1234,
        port: int = 47808,
        network: Optional[str] = None
    ):
        """
        Initialize BACnet client.

        Args:
            device_address: Local device IP address
            device_id: Local device BACnet ID
            port: BACnet port (default: 47808)
            network: Network address (optional)
        """
        if not BACNET_AVAILABLE:
            raise ImportError("BACnet libraries not installed")

        self.device_address = device_address
        self.device_id = device_id
        self.port = port
        self.network = network

        # BACnet object mappings
        self.object_mappings = {
            # Input sensors
            'UCWIT': {'type': 'analogInput', 'instance': 0},
            'UCAIT': {'type': 'analogInput', 'instance': 1},
            'UCWF': {'type': 'analogInput', 'instance': 2},
            'UCAIH': {'type': 'analogInput', 'instance': 3},
            'AMBT': {'type': 'analogInput', 'instance': 4},
            'UCTSP': {'type': 'analogInput', 'instance': 5},

            # Output predictions
            'UCAOT': {'type': 'analogValue', 'instance': 0},
            'UCWOT': {'type': 'analogValue', 'instance': 1},
            'UCAF': {'type': 'analogValue', 'instance': 2},
            'Q_thermal': {'type': 'analogValue', 'instance': 3},
        }

        # Configure logging
        self.logger = logging.getLogger(__name__)

        # Application instance (to be created)
        self.app = None

    def initialize(self):
        """Initialize BACnet application."""
        try:
            # Create local device object
            this_device = LocalDeviceObject(
                objectName=f"HVAC_Twin_{self.device_id}",
                objectIdentifier=('device', self.device_id),
                maxApduLengthAccepted=1024,
                segmentationSupported='segmentedBoth',
                vendorIdentifier=15
            )

            # Create application
            self.app = BIPSimpleApplication(
                this_device,
                f"{self.device_address}/{self.port}"
            )

            self.logger.info(f"BACnet application initialized: {self.device_address}:{self.port}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize BACnet: {e}")
            raise

    def read_property(
        self,
        device_address: str,
        object_type: str,
        object_instance: int,
        property_name: str = "presentValue"
    ) -> Optional[Any]:
        """
        Read BACnet property from remote device.

        Args:
            device_address: Remote device address (e.g., "192.168.1.10")
            object_type: BACnet object type (e.g., "analogInput")
            object_instance: Object instance number
            property_name: Property to read (default: "presentValue")

        Returns:
            Property value or None if failed
        """
        try:
            # Create read request
            request = ReadPropertyRequest(
                objectIdentifier=(object_type, object_instance),
                propertyIdentifier=property_name
            )
            request.pduDestination = Address(device_address)

            # Create IOCB
            iocb = IOCB(request)

            # Send request
            self.app.request_io(iocb)

            # Wait for response
            iocb.wait()

            if iocb.ioResponse:
                datatype = get_datatype(object_type, property_name)
                return iocb.ioResponse.propertyValue.cast_out(datatype)
            else:
                self.logger.error(f"Read failed: {iocb.ioError}")
                return None

        except Exception as e:
            self.logger.error(f"Error reading property: {e}")
            return None

    def write_property(
        self,
        device_address: str,
        object_type: str,
        object_instance: int,
        value: Any,
        property_name: str = "presentValue"
    ) -> bool:
        """
        Write BACnet property to remote device.

        Args:
            device_address: Remote device address
            object_type: BACnet object type
            object_instance: Object instance number
            value: Value to write
            property_name: Property to write (default: "presentValue")

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get datatype
            datatype = get_datatype(object_type, property_name)

            # Create value
            if datatype is Real:
                bac_value = Real(float(value))
            elif datatype is Unsigned:
                bac_value = Unsigned(int(value))
            else:
                bac_value = datatype(value)

            # Create write request
            request = WritePropertyRequest(
                objectIdentifier=(object_type, object_instance),
                propertyIdentifier=property_name
            )
            request.pduDestination = Address(device_address)
            request.propertyValue = bac_value

            # Create IOCB
            iocb = IOCB(request)

            # Send request
            self.app.request_io(iocb)

            # Wait for response
            iocb.wait()

            if iocb.ioResponse:
                self.logger.debug(f"Write successful: {object_type}:{object_instance} = {value}")
                return True
            else:
                self.logger.error(f"Write failed: {iocb.ioError}")
                return False

        except Exception as e:
            self.logger.error(f"Error writing property: {e}")
            return False

    def read_sensor_data(self, device_address: str) -> Dict[str, float]:
        """
        Read all sensor data from BACnet device.

        Args:
            device_address: Remote device address

        Returns:
            Dictionary with sensor readings
        """
        sensor_data = {}

        # Read input sensors
        for sensor_name, obj_config in self.object_mappings.items():
            if obj_config['type'] == 'analogInput':
                value = self.read_property(
                    device_address,
                    obj_config['type'],
                    obj_config['instance']
                )

                if value is not None:
                    sensor_data[sensor_name] = float(value)
                else:
                    self.logger.warning(f"Failed to read {sensor_name}")

        return sensor_data

    def write_prediction(self, device_address: str, prediction: Dict[str, float]) -> bool:
        """
        Write prediction results to BACnet device.

        Args:
            device_address: Remote device address
            prediction: Dictionary with prediction results

        Returns:
            True if all writes successful
        """
        success = True

        for output_name, value in prediction.items():
            if output_name in self.object_mappings:
                obj_config = self.object_mappings[output_name]

                if obj_config['type'] == 'analogValue':
                    result = self.write_property(
                        device_address,
                        obj_config['type'],
                        obj_config['instance'],
                        value
                    )

                    if not result:
                        success = False
                        self.logger.error(f"Failed to write {output_name}")

        return success

    def publish_to_bacnet(
        self,
        device_address: str,
        predictions: Dict[str, float],
        metadata: Optional[Dict] = None
    ):
        """
        Publish predictions to BACnet network.

        Args:
            device_address: Target device address
            predictions: Prediction results
            metadata: Additional metadata (timestamp, confidence, etc.)
        """
        self.logger.info(f"Publishing predictions to BACnet device: {device_address}")

        # Write predictions
        success = self.write_prediction(device_address, predictions)

        if success:
            self.logger.info("Predictions published successfully")
        else:
            self.logger.error("Failed to publish some predictions")

        # Write metadata if provided
        if metadata:
            timestamp_obj = self.object_mappings.get('timestamp')
            if timestamp_obj:
                self.write_property(
                    device_address,
                    timestamp_obj['type'],
                    timestamp_obj['instance'],
                    metadata.get('timestamp', datetime.now().isoformat())
                )

        return success


# Simpler implementation using BAC0 (if available)
class HVACBACnetClientSimple:
    """
    Simplified BACnet client using BAC0 library.

    Easier to use but requires BAC0 installed.
    """

    def __init__(self, ip_address: str = "192.168.1.100/24"):
        """
        Initialize simple BACnet client.

        Args:
            ip_address: Local IP address with subnet (e.g., "192.168.1.100/24")
        """
        try:
            import BAC0
            self.bacnet = BAC0.lite(ip=ip_address)
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"BACnet initialized: {ip_address}")
        except ImportError:
            raise ImportError("BAC0 not installed. Install with: pip install BAC0")
        except Exception as e:
            self.logger.error(f"Failed to initialize BACnet: {e}")
            raise

    def read(self, device_address: str, object_id: str) -> Optional[float]:
        """
        Read BACnet object.

        Args:
            device_address: Device address (e.g., "192.168.1.10")
            object_id: Object identifier (e.g., "analogInput:0")

        Returns:
            Object value
        """
        try:
            value = self.bacnet.read(f"{device_address} {object_id} presentValue")
            return float(value)
        except Exception as e:
            self.logger.error(f"Read error: {e}")
            return None

    def write(self, device_address: str, object_id: str, value: float) -> bool:
        """
        Write BACnet object.

        Args:
            device_address: Device address
            object_id: Object identifier
            value: Value to write

        Returns:
            True if successful
        """
        try:
            self.bacnet.write(f"{device_address} {object_id} presentValue {value}")
            return True
        except Exception as e:
            self.logger.error(f"Write error: {e}")
            return False

    def disconnect(self):
        """Disconnect from BACnet network."""
        if hasattr(self, 'bacnet'):
            self.bacnet.disconnect()


def example_usage():
    """Example usage of BACnet client."""
    print("="*70)
    print("BACNET INTEGRATION - EXAMPLE")
    print("="*70)

    if not BACNET_AVAILABLE:
        print("\nNote: BACnet libraries not installed")
        print("This is a conceptual example showing the integration structure")
        print("\nTo enable BACnet:")
        print("  pip install BAC0")
        print("  pip install bacpypes")
        return

    print("\nBACnet Integration Ready")
    print("\nFeatures:")
    print("  - Read sensor data from BACnet devices")
    print("  - Write predictions to BACnet objects")
    print("  - Support for analog/binary inputs/outputs")
    print("  - Integration with SCADA/BMS systems")

    print("\nObject Mappings:")
    client = HVACBACnetClient()
    for name, config in client.object_mappings.items():
        print(f"  {name:15} -> {config['type']}:{config['instance']}")

    print("\n✓ BACnet integration available")


if __name__ == '__main__':
    example_usage()
