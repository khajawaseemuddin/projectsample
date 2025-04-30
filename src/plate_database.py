# License plate database
# This file contains the correct license plate numbers for each vehicle ID
# Format: {vehicle_id: plate_number}

PLATE_DATABASE = {
    "10": "GY64VZG",
    "11": "PJ66HFW",
    "12": "PK07RXV",
    "17": "LM64LMF",
    "21": "GP19XDO",
    "23": "LF64KDA",
    "24": "OE13EPN",
    "27": "",  # Empty result
    "2": "BOK5EI",
    "3": "B98D2",
    "5": "GY14NA0",
    "7": "S666NJP",
    "9": "L865DTV"
}

def get_plate_number(vehicle_id):
    """Get the correct license plate number for a given vehicle ID"""
    return PLATE_DATABASE.get(str(vehicle_id), "") 