# Reverse mappings for attributes
ATTRIBUTES_REVERSE_MAPPINGS = {
    "insurance_type": {-1: "Unknown", 1: 'Medicare', 2: 'Medicaid', 3: 'Private', 4: 'Other'},
    "anesthesia": {-1: "Unknown", 0: 'general', 1: 'regional', 2: 'hybrid', 3: 'MAC'},
    "sex": {-1: "Unknown", 0: 'Female', 1: 'Male'},
    "ethnic_group": {-1: "Unknown or declined to state", 0: 'Non-Hispanic/Non-Latino', 1: 'Hispanic/Latino'},
    "race": {-1: "Unknown or declined to state", 0: 'White', 1: 'Other', 2: 'Asian', 3: 'Black', 4: 'Hawaiian or Pacific Islander', 5: 'Native American'},
    "race_ethnicity": {-1: "Unknown or declined to state", 0: 'White', 1: 'Asian', 2: 'Hispanic/Latino', 3: 'Non-Hispanic Other', 4: 'Black', 5: 'Hawaiian or Pacific Islander', 6: 'Native American'}
}

