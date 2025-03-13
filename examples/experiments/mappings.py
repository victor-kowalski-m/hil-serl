from experiments.ram_insertion.config import TrainConfig as RAMInsertionTrainConfig
from experiments.resistor_insertion.config import TrainConfig as ResistorInsertionTrainConfig
from experiments.usb_pickup_insertion.config import (
    TrainConfig as USBPickupInsertionTrainConfig,
)
from experiments.object_handover.config import TrainConfig as ObjectHandoverTrainConfig
from experiments.egg_flip.config import TrainConfig as EggFlipTrainConfig
from experiments.cable_route.config import TrainConfig as CableRouteTrainConfig
from experiments.pcb_insertion.config import TrainConfig as PCBInsertionTrainConfig

CONFIG_MAPPING = {
    "ram_insertion": RAMInsertionTrainConfig,
    "resistor_insertion": ResistorInsertionTrainConfig,
    "usb_pickup_insertion": USBPickupInsertionTrainConfig,
    "object_handover": ObjectHandoverTrainConfig,
    "egg_flip": EggFlipTrainConfig,
    "cable_route": CableRouteTrainConfig,
    "pcb_insertion": PCBInsertionTrainConfig,
}
