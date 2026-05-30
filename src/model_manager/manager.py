import torch
import cpuinfo
from loguru import logger as log

class ModelManager:
    def __init__(self):
        pass

    def get_training_devices(self):
        log.info("Checking training devices...")
        cuda_compatible = torch.cuda.is_available()
        log.debug(f"CUDA Detection: {cuda_compatible}")
        if cuda_compatible:
            devices = [device for device in range(torch.cuda.device_count())]
            device_names = [torch.cuda.get_device_name(device) for device in devices]
            return devices, device_names
        
        cpu_name = cpuinfo.get_cpu_info().get("brand_raw", "Name Couldn't Be Determined")
        return ["cpu"], [cpu_name]


if __name__ == '__main__':
    model_manager = ModelManager()
    device, device_name = model_manager.get_training_devices()
    print(f"Device: {device}, Device Name: {device_name}")