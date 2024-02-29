price_table = {
    'A100-SXM-80GB': 8.80,
    "NVIDIA_A100-SXM4-40GB": 2.34, # use reserved price
    "TESLA_V100-SXM2-32GB": 1.86,
    "TESLA_T4": 0.70,
}

def get_price(device_name):
    device_name = device_name.upper()
    if device_name in price_table:
        return price_table[device_name]
    else:
        raise ValueError("Unknown device name: {}".format(device_name))